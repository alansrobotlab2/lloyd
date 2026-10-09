"""The review rung, and the retry it drives.

Every other rung asks whether the change broke something. #544 passed all
eight with one acceptance clause skipped, half a fleet uncovered, a silent
`Edit` replay and an `or True` assertion — then declared itself `deferred` to
nothing and parked its item forever. What follows pins the mechanism that
would have caught it, and the loop that gives a sound-premise round another
go without letting author and grader disagree indefinitely.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, gate as G, review as RV, round as R
from scripts.automod import state as S, vault_round as V, worktree as W
from workers.sources import autocode as I
from workers.sources import autotriage as T


@pytest.fixture(autouse=True)
def _seams_first(monkeypatch):
    """This file tests the review rung's mechanics — seams by attempt,
    precheck severities, amendment handling — so it pins `seams_block: first`,
    under which a testable seam still refuses on attempt 1. The shipped
    setting is `never` (2026-09-24, `tests/test_review_grader_policy.py`).
    Set through the config `review.seams_policy` reads, not by patching it.
    (These tests ran under the `table` policy until it was retired the same
    day; they decide under the grader policy, the only one there is.)"""
    from app.config import CONFIG
    automod = dict(CONFIG.get("automod") or {})
    review = dict(automod.get("review") or {})
    review["seams_block"] = "first"
    automod["review"] = review
    monkeypatch.setitem(CONFIG, "automod", automod)


ROOT = Path(__file__).resolve().parent.parent
HONESTY_FIXTURES = ROOT / "tests" / "fixtures" / "honesty"


def _fixture(name: str) -> str:
    """A measured test file's body, read from `tests/fixtures/honesty/`.

    The bodies the honesty prechecks are measured against live there rather than
    inline here because this file is itself one of the files `honesty_prechecks`
    runs on, and the review rung computes them with the LIVE checkout's
    `review.py` (#1755) — a round that changes that checker is graded by the
    version it replaces, and the one still running counts the five dishonesty
    patterns in the raw text of every changed test file, prose and string
    literals included. A fixture spelled inline is therefore counted as newly
    dishonest code and refuses the round that added it: that is how
    `SM_20260928_210355` spent its second review attempt. A `.txt` is not a test
    file (`testpaths.is_test_file` asks for `.py`), so each shape stays readable
    in the code it is, and the node measures the real thing.
    """
    return (HONESTY_FIXTURES / name).read_text(encoding="utf-8")


def _why(rx_fragment: str) -> str:
    """The problem string the pattern table attaches to the pattern whose
    regular expression contains `rx_fragment` — looked up off the table rather
    than quoted here, for the same reason `_fixture` reads its bodies off disk.
    Pass a fragment of the REGEX as `review.py` writes it (`skip\\(`, not the
    spelling), and it is unique by construction: two matches is a table this
    file can no longer read, and the node says so."""
    hits = [why for pat, why, _sev in RV._HONESTY_PATTERNS if rx_fragment in pat]
    assert len(hits) == 1, (rx_fragment, hits, [p for p, _w, _s in RV._HONESTY_PATTERNS])
    return hits[0]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def write_item(d: Path, item_id, *, status="up_next", clauses=None, board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created, "board": board,
          "tags": ["backlog"]}
    if clauses:
        fm["acceptance_clauses"] = list(clauses)
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
                    encoding="utf-8")
    return path


def _confirm(item_id, acceptance="the check passes", clauses=(), surface=""):
    S.append_event({"event": "backlog_triage", "item_id": item_id, "verdict": "confirmed",
                    "acceptance": acceptance, "acceptance_clauses": list(clauses),
                    **({"surface": surface} if surface else {})},
                   path=S.LEDGER_PATH)


def _round(item_id, rid, *, stop_reason="stop"):
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished",
                    "round_id": rid, "stop_reason": stop_reason, "num_turns": 40},
                   path=S.LEDGER_PATH)


def _review_refused(rid, item_id, *, clauses_unmet=(1,), findings="clause 1 unmet: no test"):
    S.append_event({"event": "review", "round_id": rid, "item_id": item_id, "blocking": True,
                    "kind": "retry", "premise": "sound",
                    "clauses": [{"clause": i, "verdict": "unmet"} for i in clauses_unmet]},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": rid, "rung": "review", "ok": False,
                    "detail": f"review sent it back: {findings}", "review_retry": True,
                    "review_findings": findings}, path=S.LEDGER_PATH)


def _fm(path):
    return B._split_frontmatter(path.read_text())[0]


# ── the schema is derived, strict, and bans nothing the grader needs ────────

def test_the_review_schema_is_built_from_the_tuples_and_is_strict():
    s = RV.REVIEW_SCHEMA
    assert s["properties"]["premise"]["enum"] == list(RV.PREMISES)
    clause = s["properties"]["clauses"]["items"]
    assert clause["properties"]["verdict"]["enum"] == list(RV.CLAUSE_VERDICTS)
    assert clause["properties"]["how_verified"]["enum"] == list(RV.HOW_VERIFIED)
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    assert clause["additionalProperties"] is False and set(clause["required"]) == set(clause["properties"])
    # #2240 REVERSED the policy this node used to pin. The line here read
    # `assert "maxLength" not in json.dumps(s)` — an uncapped grammar on purpose,
    # because "the decoder would stop mid-sentence at it" — and that is what made a
    # multi-clause vault review ungradeable: with any string left open the
    # finalizer reads a grading that ran to `max_tokens` as a budget and tells its
    # reader to raise `harness.finalizer.max_tokens`, nine `vault_review` rows
    # later. So the pin stands, pointed the other way: every string this schema
    # admits a grader may write carries a positive cap. An `enum` needs none —
    # `premise`, `verdict`, `how_verified` and `severity` are bounded by their
    # value sets. The leaf-by-leaf pin, the control that capping only the clause
    # row is STILL unbounded, and the proof that six of the eight caps equal
    # `parse_review`'s own post-parse slices are all in
    # tests/test_automod_schema_bounds.py. The other verdict schemas' maxLength-absent
    # nodes (#2216's to reverse) are no longer waiting: #2444 reversed all five of them
    # — tests/test_backlog_unattended.py, test_review_policy.py,
    # test_structured_verdict.py, test_youtube_digest_source.py and
    # test_arch_review_source.py now require a cap on their schema's string leaves.
    from tests.test_automod_schema_bounds import string_leaves
    open_strings = {path: node for path, node in string_leaves(s).items()
                    if not node.get("enum")}
    assert len(open_strings) == 8, (
        f"a grader-writable string appeared or vanished: {sorted(open_strings)}")
    for path, node in open_strings.items():
        assert isinstance(node.get("maxLength"), int) and node["maxLength"] > 0, (
            f"{path} is uncapped again")


def test_the_grader_is_denied_every_automod_verb_and_every_writer():
    from workers.sources import _common as C
    assert set(C.WORKER_AUTOMOD_BAN) <= set(RV.REVIEW_DENY)
    assert set(C.WORKER_GRANT_MINT_BAN) <= set(RV.REVIEW_DENY)
    for name in ("Edit", "Write", "backlog_write_task", "vault_write", "Task"):
        assert name in RV.REVIEW_DENY
    # ...and is not denied what it exists to do.
    for name in ("Read", "Grep", "Glob", "Bash"):
        assert name not in RV.REVIEW_DENY


# ── parse_review judges the judge ──────────────────────────────────────────

def _obj(**clause):
    base = {"clause": 1, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 3,
            "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran", "note": "ok"}
    base.update(clause)
    return {"premise": "sound", "clauses": [base], "test_honesty": [], "seams_unverified": [],
            "summary": "fine"}


@pytest.fixture
def wt(tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "tests").mkdir()
    # Six lines, so `_obj`'s default `evidence_line` 3 is inside the file (#1254).
    (tmp_path / "app" / "x.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    (tmp_path / "tests" / "test_x.py").write_text("def test_it():\n    assert 1\n")
    # A test this diff did not touch, for the existing-test shape.
    (tmp_path / "tests" / "test_old.py").write_text("def test_before():\n    assert 1\n")
    return tmp_path


def test_a_met_with_real_evidence_stands(wt):
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "met" and parsed["downgraded"] == []


@pytest.mark.parametrize("tests_passed", [False, True])
@pytest.mark.parametrize("bad", [
    {"evidence_path": "app/nope.py"},
    {"evidence_path": ""},
    {"test_node_id": "tests/test_other.py::test_z"},
    {"test_node_id": ""},
    {"how_verified": "inferred"},
])
def test_a_met_without_evidence_is_downgraded_in_python(wt, bad, tests_passed):
    """The grader's laziness is not the author's pass. A `met` needs a real
    path, a test in a file this diff changed, and ran|read — or it is
    `partial`, decided here without asking the model again. A green tests
    rung waives none of these five: a missing file is still missing."""
    parsed = RV.parse_review(_obj(**bad), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=tests_passed, changed_paths=["app/x.py", "tests/test_x.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c["downgraded"] and parsed["downgraded"] == [1]
    assert "accepted" not in c


# ── the three shapes a `met` may stand on besides a changed test ─────────────

def _met(wt, *, tests_passed=True, changed_paths=("app/x.py", "tests/test_x.py"), **clause):
    parsed = RV.parse_review(_obj(**clause), worktree=wt, changed_tests=["tests/test_x.py"],
                             n_clauses=1, tests_passed=tests_passed, changed_paths=list(changed_paths))
    return parsed["clauses"][0]


def test_a_suite_level_run_stands_on_a_green_tests_rung(wt):
    """#860's clause 8 — "the autoresearch suite passes" — was refused three
    times for the only honest node it has: `tests/ -k autoresearch`."""
    c = _met(wt, test_node_id="tests/ -k autoresearch", how_verified="ran")
    assert c["verdict"] == "met" and "downgraded" not in c
    assert "suite-level run" in c["accepted"][0]


def test_a_suite_level_run_without_a_green_tests_rung_is_partial(wt):
    c = _met(wt, tests_passed=False, test_node_id="tests/ -k autoresearch", how_verified="ran")
    assert c["verdict"] == "partial"
    assert c["downgraded"] == ["test_node_id not in a test file this diff changed"]


def test_a_suite_level_run_the_grader_only_read_is_partial(wt):
    c = _met(wt, test_node_id="tests/ -k autoresearch", how_verified="read")
    assert c["verdict"] == "partial"


# ── #1322: a node under a non-root testpath is judged like one under tests/ ──

@pytest.fixture
def harness_wt(wt):
    """`wt` with the tree's own testpaths and a harness suite beside root
    `tests/`: one file the diff changed, one it did not."""
    (wt / "pytest.ini").write_text("[pytest]\ntestpaths = tests app/harness/tests scripts\n")
    h = wt / "app" / "harness" / "tests"; h.mkdir(parents=True)
    (h / "test_x.py").write_text("def test_y():\n    assert 1\n")
    (h / "test_old.py").write_text("def test_before():\n    assert 1\n")
    return wt


def _harness_met(wt, *, tests_passed=True, **clause):
    changed = ["app/harness/loop.py", "app/harness/tests/test_x.py"]
    parsed = RV.parse_review(_obj(**clause), worktree=wt, changed_tests=changed[1:], n_clauses=1,
                             tests_passed=tests_passed, changed_paths=changed)
    return parsed, parsed["clauses"][0]


def test_a_met_pinned_in_a_changed_harness_test_stands(harness_wt):
    """SM_20260921_030016: all five clauses graded met, four downgraded
    because their tests lived in app/harness/tests/."""
    parsed, c = _harness_met(harness_wt, test_node_id="app/harness/tests/test_x.py::test_y",
                             how_verified="read", tests_passed=False)
    assert c["verdict"] == "met" and "downgraded" not in c and parsed["downgraded"] == []


def test_an_unchanged_harness_node_needs_a_run_and_a_green_rung(harness_wt):
    node = "app/harness/tests/test_old.py::test_before"
    _, c = _harness_met(harness_wt, test_node_id=node, how_verified="ran")
    assert c["verdict"] == "met" and "existing test" in c["accepted"][0]
    for how, passed in (("read", True), ("ran", False)):
        _, c = _harness_met(harness_wt, test_node_id=node, how_verified=how, tests_passed=passed)
        assert c["verdict"] == "partial"
        assert c["downgraded"] == ["test_node_id not in a test file this diff changed"]
    # `scripts` is a testpath pytest walks, not a test tree: its code is no node.
    (harness_wt / "scripts" / "automod").mkdir(parents=True)
    (harness_wt / "scripts" / "automod" / "review.py").write_text("def f():\n    pass\n")
    _, c = _harness_met(harness_wt, test_node_id="scripts/automod/review.py::f", how_verified="ran")
    assert c["verdict"] == "partial"


def test_an_existing_test_outside_the_diff_stands_when_it_was_run(wt):
    """#487's clause 4 named a real test file the round did not change."""
    c = _met(wt, test_node_id="tests/test_old.py::test_before", how_verified="ran")
    assert c["verdict"] == "met" and "existing test" in c["accepted"][0]
    assert _met(wt, test_node_id="tests/test_old.py", how_verified="ran")["verdict"] == "met"


# ── a test that fails at base too is not evidence for this diff ──────────────

@pytest.mark.parametrize("node, how", [
    ("tests/test_old.py::test_before", "ran"),     # the failing node itself
    ("tests/test_old.py::test_before", "read"),
    ("tests/test_old.py", "ran"),                  # a file-level run of its file
    ("tests/test_old.py::test_other", "ran"),      # a sibling in the same red file
])
def test_a_met_on_a_node_that_fails_at_base_does_not_stand(wt, node, how):
    """Since 2026-09-24 the tests rung passes over failures that reproduce at
    the round's base. A clause resting on one was verified by nothing: the
    rung passed over it, not through it — whatever `how` says."""
    parsed = RV.parse_review(_obj(test_node_id=node, how_verified=how), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"],
                             pre_existing_failures={"tests/test_old.py::test_before"})
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", c
    assert any("fails at base" in w for w in c["downgraded"]), c
    # The counterfactual: without the list, the same citation stands (ran) —
    # so the list, not something else, is what refused it.
    if how == "ran":
        assert _met(wt, test_node_id=node, how_verified="ran")["verdict"] == "met"


def test_a_changed_test_beside_an_unrelated_red_file_still_stands(wt):
    """Only a citation INTO the red set is refused; the diff's own test holds."""
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"],
                             pre_existing_failures={"tests/test_old.py::test_before"})
    assert parsed["clauses"][0]["verdict"] == "met"


def test_the_grader_is_told_which_failures_predate_the_round(tmp_path):
    contract = {"id": 1, "title": "t", "body": "b", "clauses": ["c"]}
    kw = dict(contract=contract, diff="", diff_truncated=False, changed_tests=[],
              test_counts={"passed": 5}, worktree=tmp_path, run_tests=tmp_path / "rt")
    plain = RV.build_prompt(**kw)
    told = RV.build_prompt(**kw, pre_existing_failures=["tests/test_uptake.py::test_a"])
    assert "fail here AND at the round's base" not in plain
    assert "tests/test_uptake.py::test_a" in told and "at most `partial`" in told
    # With nothing pre-existing the prompt is byte-identical to before.
    assert plain == RV.build_prompt(**kw, pre_existing_failures=[])


# ── #2177: a bare function name is a node too, when a changed test defines it ─

def test_a_bare_node_name_a_changed_test_defines_stands_and_names_the_file(wt):
    """Round SM_20261004_084454: the grader graded all six clauses `met`, named a
    node in `tests/test_compaction.py` for each, and every one was downgraded
    `test_node_id not in a test file this diff changed` — because it wrote the
    function name and not `path::name`, and the rail read the text before the `::`
    that was not there as a path. `REVIEW_SCHEMA` never asked for the path form, so
    the refusal condemned an honest citation and the round could not pass however
    good its diff was.

    A bare name a changed test file DEFINES now resolves into that file and holds on
    the same footing as the path-qualified spelling of the same node: `read` is
    enough, because the `tests` rung ran that file either way. And the resolution is
    recorded under `accepted`, naming the file, so a clause that stands on a
    completed citation is visible as doing so.
    """
    # The whole sentence, not a substring of it: a waiver that stopped naming the
    # node it resolved, or the file it resolved into, is a waiver nobody can audit.
    waiver = "bare node `test_it` resolved to tests/test_x.py, a test file this diff changed"
    for how in ("ran", "read"):
        c = _met(wt, test_node_id="test_it", how_verified=how)
        assert c["verdict"] == "met", (how, c)
        assert "downgraded" not in c
        assert c["accepted"] == [waiver], (how, c)
    # The rail answers the same way when asked directly, with no parse_review between.
    holds, reason = RV._node_rail("test_it", worktree=wt, changed={"tests/test_x.py"},
                                  how="ran", tests_passed=True)
    assert holds and reason == waiver, reason
    # A node already spelled with its path keeps its waiver-free `met`: the record
    # says nothing happened here, because nothing did.
    holds, reason = RV._node_rail("tests/test_x.py::test_it", worktree=wt,
                                  changed={"tests/test_x.py"}, how="ran", tests_passed=True)
    assert holds and reason == "", reason


def test_a_bare_name_resolves_only_when_exactly_one_changed_test_defines_it(wt):
    """Two changed test files, one definition, resolves into the file that has it;
    two definitions resolve nowhere, because a name two files answer to pins no
    single test — and pinning one is the entire purpose of a node id."""
    (wt / "tests" / "test_y.py").write_text("def test_second():\n    assert 1\n")
    both = ["tests/test_x.py", "tests/test_y.py"]

    def judge(node):
        parsed = RV.parse_review(_obj(test_node_id=node, how_verified="ran"), worktree=wt,
                                 changed_tests=both, n_clauses=1, tests_passed=True,
                                 changed_paths=["app/x.py"] + both)
        return parsed["clauses"][0]

    c = judge("test_second")
    assert c["verdict"] == "met" and "tests/test_y.py" in c["accepted"][0], c
    (wt / "tests" / "test_x.py").write_text("def test_second():\n    assert 1\n")
    c = judge("test_second")
    assert c["verdict"] == "partial", c
    assert c["downgraded"] == ["test_node_id not in a test file this diff changed"], c


def test_a_bare_name_resolves_through_the_changed_list_the_gate_builds(harness_wt):
    """The seam this fix actually crosses: the grader answers in another process, and
    the gate hands `parse_review` `changed_tests = TP.pick_test_files(changed,
    worktree)` (scripts/automod/gate.py) as the resolver's whole search set. So the
    resolution is built from that list and from `pytest.ini`, never from a `tests/`
    prefix of its own — which is why a bare id whose definition lives in
    `app/harness/tests/`, the tree #1322 had to teach this rail to see at all,
    resolves there as readily as one under root `tests/`."""
    changed = ["app/harness/loop.py", "app/harness/tests/test_x.py", "tests/test_x.py"]
    parsed = RV.parse_review(_obj(test_node_id="test_y", how_verified="ran"),
                             worktree=harness_wt,
                             changed_tests=RV.TP.pick_test_files(changed, harness_wt),
                             n_clauses=1, tests_passed=True, changed_paths=changed)
    c = parsed["clauses"][0]
    assert c["verdict"] == "met", c
    assert "app/harness/tests/test_x.py" in c["accepted"][0], c


@pytest.mark.live_vault
def test_the_bare_node_ids_the_rung_refused_have_history_behind_them():
    """The bytes behind this item's quoted report, put where they get read. #2168's
    clause-5 half was refused twice for exactly this shape: findings that quote a line
    range of a session file that gets swept with the runtime logs. The review session
    whose six `test_node_id` values this rail refused is now in
    `backlog/data/20261004_022144_review_3345.json` — 2,879 lines of one JSON session
    document — so the refusal's input is citable in git and the claim can be re-run
    against the real object instead of a quoted transcript.

    Read through the vault pointer the loop itself uses, so what is asserted is what
    the next reader finds on disk. The six are the item's whole claim: bare ids, all
    `how_verified: ran`, from the review of #2168's round SM_20261004_084454.
    """
    witness = V.VAULT / "backlog" / "data" / "20261004_022144_review_3345.json"
    if not witness.is_file():
        pytest.skip(f"{witness} is not on disk, so the vault copy is unreadable here")
    sess = json.loads(witness.read_text(errors="replace"))
    assert sess["title"] == "review #2168 (SM_20261004_084454)", sess["title"]
    graded = [m["structured"] for m in sess["messages"] if m.get("structured")]
    assert len(graded) == 1, f"expected one structured verdict, got {len(graded)}"
    clauses = graded[0]["clauses"]
    bare = [c for c in clauses
            if isinstance(c.get("test_node_id"), str) and c["test_node_id"]
            and "::" not in c["test_node_id"]]
    assert len(clauses) == 6 and len(bare) == 6, (
        f"the item quotes six bare ids, got {len(bare)} bare of {len(clauses)} clauses")
    assert all(c["verdict"] == "met" and c.get("how_verified") == "ran" for c in bare), bare
    assert "test_the_sidecar_is_keyed_by_session_id" in [c["test_node_id"] for c in bare]


def test_the_live_rung_and_the_replay_parser_now_agree_on_a_bare_node(wt):
    """The second boundary this fix sits across: a graded clause is written to the
    ledger here and read back later by `review_tools.parsed_from_event`, which since
    before this round has forgiven exactly this clause shape — `::`-less node, `ran` —
    and restored it to `met`. That is why the same clause reads as met wherever the
    ledger is replayed — 32 events across 29 rounds carry the downgrade — while the
    gate that wrote those events refused them: one surface blessed the shape, the
    other punished it. After the fix the live rung reaches `met`
    on its own, so the replay needs no forgiveness to agree — `approximated == 0` is
    the assertion, not the verdict, since the verdict alone would read the same had the
    leniency silently done the work again.

    The pre-fix event is replayed too: history keeps reading as met. Both directions
    have to hold or the scorecards and the gate start disagreeing about the PAST, which
    is the same disagreement with a date on it.
    """
    from scripts.automod import review_tools as RT
    live = RV.parse_review(_obj(test_node_id="test_it", how_verified="ran"), worktree=wt,
                           changed_tests=["tests/test_x.py"], n_clauses=1, tests_passed=True,
                           changed_paths=["app/x.py", "tests/test_x.py"])
    clause = live["clauses"][0]
    assert clause["verdict"] == "met" and "downgraded" not in clause, clause
    replayed, approximated = RT.parsed_from_event(
        {"clauses": [dict(clause)], "premise": "sound", "summary": "ok"})
    assert replayed["clauses"][0]["verdict"] == "met", replayed
    assert approximated == 0, "the replay forgave it again, so the two surfaces are " \
                              "still reaching `met` by different routes"
    pre_fix = {"clauses": [{"clause": 1, "verdict": "partial",
                            "downgraded": ["test_node_id not in a test file this diff changed"],
                            "test_node_id": "test_it", "how_verified": "ran", "note": ""}],
               "premise": "sound", "summary": "ok"}
    replayed, approximated = RT.parsed_from_event(pre_fix)
    assert replayed["clauses"][0]["verdict"] == "met" and approximated == 1, replayed


_UNRESOLVABLE_TREE = (
    "def test_it():\n    assert 1\n\n"
    "def helper_only():\n    return 2\n\n"
    "class TestThing:\n    def test_method(self):\n        assert 1\n\n"
    "# the case that reds is test_only_mentioned, per the item\n")


@pytest.mark.parametrize("node, why_not_a_test_name", [
    ("helper_only", "a module-level def, but pytest never collects that name"),
    ("TestThing", "a class name, not a function"),
])
def test_a_bare_name_that_is_not_a_test_function_name_is_never_resolved(wt, node,
                                                                        why_not_a_test_name):
    """The resolver's first step is the name, and it is a step before any file is
    opened: `_BARE_NODE_NAME_RX` admits only a bare `test_…` identifier, so these two
    shapes are refused on their spelling and no changed file is read looking for a
    definition they could never have had. The refusal the author sees is the same
    sentence either way — this node exists so the two steps stay separately pinned,
    and a loosened name filter that started opening files for any identifier would
    red here rather than quietly widen the rail (#2177 clause 2)."""
    (wt / "tests" / "test_x.py").write_text(_UNRESOLVABLE_TREE)
    parsed = RV.parse_review(_obj(test_node_id=node, how_verified="ran"), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", (node, why_not_a_test_name, c)
    assert c["downgraded"] == ["test_node_id not in a test file this diff changed"], c


@pytest.mark.parametrize("node, where_it_fails", [
    ("test_before", "defined, but in tests/test_old.py, which this diff did not touch"),
    ("test_never_written", "defined nowhere in the tree"),
    ("test_only_mentioned", "only named in a comment in the changed file, never defined"),
    ("test_method", "a method: its node id is tests/test_x.py::TestThing::test_method"),
])
def test_a_bare_node_name_no_changed_test_defines_is_still_downgraded(wt, node, where_it_fails):
    """The leniency is a resolver, not a widening. These four names all LOOK like test
    functions, so each one reaches the definition search and loses it: resolution is by
    a module-level `def` in a file this diff changed, and a name merely mentioned, a
    name in a file the diff never touched, a name nowhere in the tree, and an indented
    method whose real node id carries a class segment all still refuse with the reason
    this item was filed for (#2177 clause 2)."""
    (wt / "tests" / "test_x.py").write_text(_UNRESOLVABLE_TREE)
    parsed = RV.parse_review(_obj(test_node_id=node, how_verified="ran"), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", (node, where_it_fails, c)
    assert c["downgraded"] == ["test_node_id not in a test file this diff changed"], (node, c)
    assert "accepted" not in c, (node, c)


def test_a_changed_test_file_the_tree_does_not_hold_resolves_nothing(wt):
    """The changed list and the tree are chosen by different code in the gate —
    `changed_tests` comes from `TP.pick_test_files` over the round worktree, while the
    review grades inside a prefetch tree checked out at the reviewed sha — so a listed
    file can be absent from the tree the resolver opens (deleted by the diff, or a path
    the prefetch did not carry). That is an unreadable file, not a definition: the
    lookup skips it and the clause refuses with the changed-file reason. It must not
    raise, and above all it must not resolve a name to a file whose bytes nobody read.
    """
    (wt / "tests" / "test_gone.py").write_text("def test_it():\n    assert 1\n")
    (wt / "tests" / "test_x.py").unlink()          # listed as changed, absent on disk
    parsed = RV.parse_review(_obj(test_node_id="test_it", how_verified="ran"), worktree=wt,
                             changed_tests=["tests/test_x.py", "tests/test_gone.py"],
                             n_clauses=1, tests_passed=True,
                             changed_paths=["app/x.py", "tests/test_x.py",
                                            "tests/test_gone.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "met" and "tests/test_gone.py" in c["accepted"][0], c
    (wt / "tests" / "test_gone.py").unlink()       # now neither listed file is readable
    parsed = RV.parse_review(_obj(test_node_id="test_it", how_verified="ran"), worktree=wt,
                             changed_tests=["tests/test_x.py", "tests/test_gone.py"],
                             n_clauses=1, tests_passed=True,
                             changed_paths=["app/x.py", "tests/test_x.py",
                                            "tests/test_gone.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", c
    assert c["downgraded"] == ["test_node_id not in a test file this diff changed"], c


def test_a_bare_name_resolving_into_a_red_file_carries_the_pre_existing_reason(wt):
    """Bare-name resolution is applied BEFORE the pre-existing veto, on both sides of
    it, or the leniency eats the veto: that veto compares node ids against full
    `path::name` strings, so a bare name never matched one, and a clause resting on a
    file the `tests` rung only passed OVER would start standing as met — verified by
    a test that fails at base with this diff absent.

    Two things are asserted, not one: the veto fires, and it fires with the
    PRE-EXISTING reason. The changed-file reason would say the opposite thing — that
    no test could be located — on a node that was just located successfully.
    """
    (wt / "tests" / "test_x.py").write_text(
        "def test_it():\n    assert 1\n\n\ndef test_already_red():\n    assert 0\n")
    red = {"tests/test_x.py::test_already_red"}
    assert RV._node_rail("test_it", worktree=wt, changed={"tests/test_x.py"}, how="ran",
                         tests_passed=True, pre_existing=red)[0] is False
    parsed = RV.parse_review(_obj(test_node_id="test_it", how_verified="ran"), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1,
                             tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"],
                             pre_existing_failures=red)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", c
    assert c["downgraded"] == ["test_node_id cites a test that fails at base too "
                               "(pre-existing, not this diff's)"], c
    assert not any("not in a test file" in d for d in c["downgraded"]), c
    assert "accepted" not in c
    # The counterfactual: the red list, and nothing else, is what refused it.
    assert _met(wt, test_node_id="test_it",
                how_verified="ran")["verdict"] == "met"


@pytest.mark.parametrize("node", ["tests/test_gone.py::test_x", "tests/test_gone.py",
                                  "app/x.py::f", "pytest -k autoresearch"])
def test_a_node_that_is_not_a_real_tests_path_is_partial(wt, node):
    assert _met(wt, test_node_id=node, how_verified="ran")["verdict"] == "partial"


def test_evidence_of_a_deleted_file_stands_beside_a_changed_test(wt):
    """#487's clause 1 was about a report the change removed, and its only
    evidence was the file's absence."""
    c = _met(wt, evidence_path="scripts/dead_report.py", test_node_id="tests/test_x.py::test_it",
             changed_paths=("scripts/dead_report.py", "tests/test_x.py"))
    assert c["verdict"] == "met" and c["accepted"][0].startswith("evidence of absence")
    # The shape #487's grader actually wrote (a made-up name, so a real file
    # in this machine's vault cannot make it pass for the wrong reason). Since
    # #1252 the second token of that citation is tried, and it is a real test
    # file — so the path rail holds on the test path and the absence waiver is
    # no longer what carries the clause. `met` either way, now for the better
    # reason: a file on disk is pointed at.
    c = _met(wt, evidence_path="~/obsidian/memory/no-such-report-7f3a.md (absent); "
                               "tests/test_x.py:396-406",
             test_node_id="tests/test_x.py::test_it")
    assert c["verdict"] == "met" and c["evidence_path"] == "tests/test_x.py"
    assert "accepted" not in c


def test_an_absence_marker_does_not_stand_without_a_node_of_its_own(wt):
    c = _met(wt, evidence_path="scripts/dead_report.py (deleted)", test_node_id="")
    assert c["verdict"] == "partial"
    # ...nor stacked on a suite-level waiver: that pins nothing.
    c = _met(wt, evidence_path="scripts/dead_report.py (deleted)",
             test_node_id="tests/ -k report", how_verified="ran")
    assert c["verdict"] == "partial"
    assert c["downgraded"][0].startswith("evidence_path missing")
    # ...and a path the diff never touched, with no marker, is just missing.
    assert _met(wt, evidence_path="scripts/dead_report.py",
                test_node_id="tests/test_x.py::test_it")["verdict"] == "partial"


def test_a_clause_the_grader_did_not_mention_is_not_met(wt):
    parsed = RV.parse_review(_obj(), worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=3)
    assert [c["verdict"] for c in parsed["clauses"]] == ["met", "partial", "partial"]
    assert parsed["clauses"][1]["note"] == "not addressed by the grader"
    # One readable index is an abstention about the CHANGE. The rung must keep
    # refusing on it, or a grader that answers one clause of three passes by
    # saying nothing about the other two.
    assert "clauses_unreadable" not in parsed


def test_clause_entries_in_another_shape_are_recorded_as_unreadable(wt):
    """`id`/`status` instead of `clause`/`verdict`: not one index is usable.

    #1443's three rounds reached this with the grader approving every clause.
    The synthesized partials stay in the result — the caller decides what they
    mean — but the parse boundary records that no index was readable, with the
    key names it actually found, so a rail failure stops reading as a verdict.
    """
    alias = {"premise": "sound", "summary": "APPROVE", "test_honesty": [],
             "seams_unverified": [],
             "clauses": [{"id": i, "status": "met", "evidence_path": "app/x.py",
                          "test_node_id": "tests/test_x.py::test_it",
                          "how_verified": "ran", "note": "graded"} for i in (1, 2)]}
    parsed = RV.parse_review(alias, worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=2)
    assert [c["verdict"] for c in parsed["clauses"]] == ["partial", "partial"]
    unread = parsed["clauses_unreadable"]
    assert unread["entries"] == 2
    assert unread["keys"] == ["evidence_path", "how_verified", "id", "note",
                              "status", "test_node_id"], unread["keys"]


def test_clause_entries_that_are_not_objects_are_still_unreadable(wt):
    """A grader that answered `["met", "met"]` graded nothing it can be asked."""
    parsed = RV.parse_review({"premise": "sound", "clauses": ["met", "met"]},
                             worktree=wt, changed_tests=[], n_clauses=2)
    assert parsed["clauses_unreadable"]["entries"] == 2
    assert parsed["clauses_unreadable"]["keys"] == []


def test_no_clause_entries_at_all_is_an_abstention_not_an_unreadable_shape(wt):
    """`clauses: []` is the grader declining to grade; it refuses as it always did.

    The unreadable flag exists for a verdict that cannot be read, so an empty
    list must not borrow it — that would let an abstention skip the attempt.
    """
    parsed = RV.parse_review(_obj(clause=1) | {"clauses": []}, worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    assert "clauses_unreadable" not in parsed
    assert [c["verdict"] for c in parsed["clauses"]] == ["partial"]


def test_the_vault_shape_needs_no_test_node(wt):
    parsed = RV.parse_review(_obj(test_node_id=""), worktree=wt, changed_tests=[], n_clauses=1,
                             require_tests=False)
    assert parsed["clauses"][0]["verdict"] == "met"


def test_an_unusable_object_is_none(wt):
    assert RV.parse_review("nope", worktree=wt, changed_tests=[], n_clauses=1) is None
    assert RV.parse_review({"premise": "maybe"}, worktree=wt, changed_tests=[], n_clauses=1) is None


def test_decide_maps_premise_then_findings():
    base = {"premise": "sound", "clauses": [{"clause": 1, "verdict": "met", "note": ""}],
            "test_honesty": [], "seams_unverified": [], "summary": "good", "downgraded": []}
    assert RV.decide(base, []) == ("pass", "good")
    assert RV.decide({**base, "premise": "unsound", "summary": "premise is false"}, [])[0] == "unsound"
    kind, findings = RV.decide({**base, "clauses": [{"clause": 2, "verdict": "unmet", "note": "no path",
                                                     "downgraded": []}]}, [])
    assert kind == "retry" and "clause 2 unmet: no path" in findings
    kind, findings = RV.decide(base, [{"file": "tests/t.py", "line": 9, "problem": "`or True`"}])
    assert kind == "retry" and "tests/t.py:9" in findings
    kind, findings = RV.decide({**base, "seams_unverified": ["loopback POST"]}, [])
    assert kind == "retry" and "seam unverified: loopback POST" in findings


# ── the deterministic honesty checks ───────────────────────────────────────

@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"; (r / "tests").mkdir(parents=True); (r / "app").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0\n")
    (r / "app" / "m.py").write_text("V = 1\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    return r, base


def test_prechecks_flag_only_what_the_round_added(repo):
    r, base = repo
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0\n\n"
        "def test_new():\n    assert rows() == [] or True\n")
    git(r, "commit", "-qam", "round")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    problems = [o["problem"] for o in out]
    assert any("or True" in p for p in problems), out
    assert not any("xfail" in p for p in problems), "the pre-existing xfail is not this round's"
    assert out[0]["line"] == 8


def test_prechecks_notice_a_code_change_with_no_new_test(repo):
    r, base = repo
    (r / "tests" / "test_a.py").write_text(
        "import pytest\n\n@pytest.mark.xfail\ndef test_old():\n    assert 0  # touched\n")
    (r / "app" / "m.py").write_text("V = 2\n")
    git(r, "commit", "-qam", "round")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=1)
    assert any("no test function was added" in o["problem"] for o in out)
    assert RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=0) == []


# ── the constant-mirroring assertion (#1678) ───────────────────────────────
# Pocock's first shape, and the one a deterministic check CAN pin: an
# assertion comparing a call's result against a constant defined at the top of
# the SAME test file. The five literal patterns spell "cannot fail" the way a
# person writes it when they give up; a mirrored constant never gives up, it
# just moves with the code, and the only thing that has ever caught it is the
# token-priced `test_honesty` grader. The ledger for 2026-09-11→09-27 carries
# 708 events with a grader-authored `test_honesty` list against 73 carrying a
# deterministic precheck line (9.7x), and 0 of the 1965 recorded entries name
# one of the five patterns' own problem strings. This moves one shape out of
# the priced half and into the free one.


def test_a_docstring_that_quotes_itself_hides_no_real_skip(tmp_path):
    """Why the blank pass runs on `tokenize` and not on a char-wise quote scan:
    a docstring holding a lone double quote plus an apostrophe desynchronises
    the scan's idea of where the literal ends, the flipped state runs to the end
    of the file, and what happens next depends on quote parity. At
    `tests/_live_data.py` the first version of the pass blanked that file's real
    unconditional skip at line 136 out of the skip pattern's reach — the shape
    that refuses a skipless round, invisible in exactly the files that skip
    most. The fixture is that shape: a docstring quoting itself, then one
    unconditional skip below it, which must still be found, at its own line."""
    r, base = _delta_repo(
        tmp_path, tag="parity", base_src=_MIRROR_BASE,
        post_src=_fixture("quote_heavy_docstring_skip.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    hits = [o for o in out if "pytest.skip" in o["problem"]]
    assert len(hits) == 1, out
    assert hits[0]["severity"] == "blocking", "an unconditional skip still refuses"
    assert hits[0]["line"] == 4, "the skip is code below the docstring, not prose in it"


def test_prose_that_looks_like_code_binds_no_constant(tmp_path):
    """An ALL_CAPS assignment written inside a docstring is documentation, not
    a binding, so the assertion below it compares against nothing the file
    defines. The detector reads the blanked text for exactly this reason — and
    the control at the end of this node proves the silence is the blanking and
    not a dead detector: the same assertion under a real module-level binding
    fires."""
    r, base = _delta_repo(
        tmp_path, tag="prosecode", base_src=_MIRROR_BASE,
        post_src='"""\n'                                                      # 1
                 'SOME_LIMIT = 5   documented here, bound nowhere\n'          # 2
                 '"""\n'                                                      # 3
                 '\n'                                                         # 4
                 'def test_the_docstring_one():\n'                            # 5
                 '    assert count(open_body()) == SOME_LIMIT\n')             # 6
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                         n_clauses=1)) == []
    r2, base2 = _delta_repo(
        tmp_path, tag="prosectl", base_src=_MIRROR_BASE,
        post_src='SOME_LIMIT = 5\n'                                           # 1
                 '\n'                                                         # 2
                 'def test_the_bound_one():\n'                                # 3
                 '    assert count(open_body()) == SOME_LIMIT\n')             # 4
    found = _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4 and "line 1" in found[0]["problem"], found


def test_a_comment_beside_an_assertion_is_not_a_second_comparison(tmp_path):
    """A trailing `# was parse(y) == EXPECTED_CHAR` puts a second `==` at depth
    zero on the same physical line as a real assertion. Un-commented, that
    comment supplies both the call and the constant name, and the round is
    blamed for an assertion nobody wrote."""
    r, base = _delta_repo(
        tmp_path, tag="comment", base_src=_MIRROR_BASE,
        post_src="EXPECTED_CHAR = 'x'\n"                                      # 1
                 '\n'                                                         # 2
                 'def test_ok():\n'                                           # 3
                 "    assert parse(open_body()) == 'x'  # was parse(y) == EXPECTED_CHAR\n")
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                         n_clauses=1)) == []
    # Control: the same file with the assertion comparing against the name the
    # comment only mentioned is exactly the thing being reported.
    r2, base2 = _delta_repo(
        tmp_path, tag="commentctl", base_src=_MIRROR_BASE,
        post_src="EXPECTED_CHAR = 'x'\n"                                      # 1
                 '\n'                                                         # 2
                 'def test_ok():\n'                                           # 3
                 "    assert parse(open_body()) == EXPECTED_CHAR\n")          # 4
    found = _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4 and "EXPECTED_CHAR" in found[0]["problem"], found


def test_an_assertion_written_over_three_lines_is_still_one_comparison(tmp_path):
    """The call and the compared name on different physical lines: a
    line-by-line read sees an unclosed `(` on the assert line and no comparison
    at all, so the mirror goes unreported. The statement is joined to Python's
    bracket rule and reported at the `assert` line."""
    r, base = _delta_repo(
        tmp_path, tag="split", base_src=_MIRROR_BASE,
        post_src="EXPECTED_TITLE = 'widget'\n"      # 1
                 "\n"                               # 2
                 "def test_it():\n"                 # 3
                 "    assert parse(\n"              # 4
                 "        open_body()\n"            # 5
                 "    ) == EXPECTED_TITLE\n")       # 6
    found = _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4, "the assertion's own line, not the closing bracket's"
    assert "EXPECTED_TITLE" in found[0]["problem"] and "line 1" in found[0]["problem"]


def _delta_repo(tmp_path, *, base_src, post_src, pytest_ini=None, new_files=None,
                tag="dr"):
    """A repo whose base commit holds `base_src` and whose HEAD holds
    `post_src` (plus `new_files`), so a delta is measured across real git
    objects rather than a stubbed `_git`. `tag` names the subdirectory, because
    one test measures several base/HEAD pairs side by side."""
    r = tmp_path / tag
    (r / "tests").mkdir(parents=True)
    (r / "app").mkdir()
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com")
    git(r, "config", "user.name", "t")
    (r / "tests" / "test_a.py").write_text(base_src)
    (r / "app" / "m.py").write_text("def parse(s):\n    return s\n")
    if pytest_ini:
        (r / "pytest.ini").write_text(pytest_ini)
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    (r / "tests" / "test_a.py").write_text(post_src)
    for rel, src in (new_files or {}).items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "round")
    return r, base


_MIRROR_BASE = "import pytest\n\n\ndef test_old():\n    assert parse('a') == 'a'\n"


def _mirrors(out):
    return [o for o in out if "mirrors" in o["problem"]]


def test_prechecks_flag_an_assertion_that_mirrors_its_own_module_constant(tmp_path):
    """Clause 1: the finding names the constant's line AND the assertion's
    line, and says the test mirrors the value instead of pinning it."""
    r, base = _delta_repo(
        tmp_path, base_src=_MIRROR_BASE,
        post_src="from app.m import parse\n"                      # 1
                 "\n"                                             # 2
                 "EXPECTED_TITLE = 'widget'\n"                    # 3
                 "\n"                                             # 4
                 "def test_title_is_the_module_default():\n"      # 5
                 "    assert parse(open_body()) == EXPECTED_TITLE\n")   # 6
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    found = _mirrors(out)
    assert len(found) == 1, out
    assert found[0]["file"] == "tests/test_a.py"
    assert found[0]["line"] == 6, "the assertion's own line"
    assert "EXPECTED_TITLE" in found[0]["problem"]
    assert "line 3" in found[0]["problem"], "and the line the constant is bound on"
    assert "instead of pinning" in found[0]["problem"], found[0]["problem"]


def test_prechecks_stay_silent_on_an_imported_name_or_an_inline_literal(tmp_path):
    """Clause 2: the ordinary way to assert a public contract must not fire.
    A name imported from the module under test is not a value the test wrote,
    and neither is a literal typed into the assertion itself — even when the
    same file also defines a constant holding that same value. The control adds
    the fourth shape to that same file — the file's own constant compared against
    a call — and the run reports exactly that one, so the three silences above
    are selective and not a detector that never fires."""
    silent = ("from app.m import MAX_ITEMS, parse\n"               # 1
              "\n"                                                 # 2
              "EXPECTED_TITLE = 'widget'\n"                        # 3
              "\n"                                                 # 4
              "def test_the_imported_contract_value():\n"          # 5
              "    assert parse(open_body()) == MAX_ITEMS\n"       # 6  imported
              "\n"                                                 # 7
              "def test_the_inline_literal():\n"                   # 8
              "    assert parse(open_body()) == 'widget'\n"        # 9  literal
              "\n"                                                 # 10
              "def test_a_value_the_test_handed_round():\n"        # 11
              "    parsed = 'widget'\n"                            # 12
              "    assert parsed == EXPECTED_TITLE\n")             # 13  no call
    r, base = _delta_repo(tmp_path, base_src=_MIRROR_BASE, post_src=silent)
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    assert _mirrors(out) == [], out
    r2, base2 = _delta_repo(
        tmp_path, tag="ctl", base_src=_MIRROR_BASE,
        post_src=silent
                 + "\n"                                            # 14
                 "def test_the_mirror_control():\n"                # 15
                 "    assert parse(open_body()) == EXPECTED_TITLE\n")   # 16
    found = _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"], n_clauses=2))
    assert len(found) == 1, found
    assert found[0]["line"] == 16 and "EXPECTED_TITLE" in found[0]["problem"], found


def test_a_bracket_in_a_string_or_a_filter_inside_a_call_moves_nothing(tmp_path):
    """The scan reads the assertion the way Python reads it, not the way a line
    grep does. A bracket that lives inside a string literal does not open an
    argument list, so the assertion after it still splits and still fires; a
    comparison buried inside a call is not what the `assert` compares, so it
    fires nothing. One finding, at line 4, is both halves at once."""
    r, base = _delta_repo(
        tmp_path, base_src=_MIRROR_BASE,
        post_src="EXPECTED_CHAR = '('\n"                          # 1
                 "\n"                                             # 2
                 "def test_the_bracket_is_a_string():\n"          # 3
                 "    assert parse('(') == EXPECTED_CHAR\n"       # 4  fires
                 "\n"                                             # 5
                 "def test_a_filter_is_not_the_assertion():\n"    # 6
                 "    assert sum(1 for t in titles if t == EXPECTED_CHAR) == 0\n")
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1)
    found = _mirrors(out)
    assert len(found) == 1, out
    assert found[0]["line"] == 4, found
    assert "EXPECTED_CHAR" in found[0]["problem"] and "line 1" in found[0]["problem"]


def _seed_repo(tag, tmp_path, seed_line):
    """A file that seeds a constant and then asserts a call still returns it,
    with the seed written however the caller wants — used by the #1864 nodes to
    vary only the way the constant enters a call."""
    return _delta_repo(
        tmp_path, tag=tag, base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '\n'                                                       # 2
                 'def _seed(target):\n'                                     # 3
                 f'    {seed_line}\n'                                       # 4
                 '\n'                                                       # 5
                 'def test_a_refused_write_left_the_file_alone():\n'        # 6
                 '    assert read_back() == PAYLOAD\n')                     # 7


def test_a_seed_constant_written_by_a_fixture_elsewhere_is_no_mirror(tmp_path):
    """#1864 clauses 1 and 2: a constant the file hands to a call as a whole
    argument is that file's own seed data, and the assertion re-reading it after
    a refused operation is the contract — it is not a mirror.

    Both false positives of 2026-09-29 have this shape, and
    `tests/test_builtin_fs_protected_write.py` is the hard one: none of its five
    firing tests calls `write_text(ORIGINAL)` itself — the writes live in the
    `home()` and `linked_home()` fixtures — so only a FILE-scope reading silences
    it, and that is what the seed call in `_seed` above is for: a different
    function from the assertion. The keyword form (`content=PAYLOAD`) is the same
    act. Then the three controls, each of which keeps the detector alive: a
    `PAYLOAD` that never enters any argument list is `FALLBACK_LAYOUT` and fires
    once, at the assertion's own line, naming the constant's; `assert
    parse(PAYLOAD) == PAYLOAD` fires, because counting the input side of the
    round-trip-that-cannot-fail as a seed would silence it — which a def
    parameter default (`def helper(n=PAYLOAD)`) did do until #1976, the one shape
    that silenced the round-trip in the corpus; and a seed call that can only take `PAYLOAD + "!"` fires too,
    because a constant fed into arithmetic is being used as an expectation, not
    handed over as one.
    """
    r, base = _seed_repo("seedpos", tmp_path, "target.write_text(PAYLOAD)")
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                         n_clauses=1)) == []

    r2, base2 = _seed_repo("seedkw", tmp_path, "target.write(content=PAYLOAD)")
    assert _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"],
                                         n_clauses=1)) == []

    r3, base3 = _delta_repo(
        tmp_path, tag="seedctl", base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '\n'                                                       # 2
                 'def test_the_constant_that_is_never_an_input():\n'        # 3
                 '    assert read_back() == PAYLOAD\n')                     # 4
    found = _mirrors(RV.honesty_prechecks(r3, base3, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4, found
    assert "PAYLOAD" in found[0]["problem"] and "line 1" in found[0]["problem"]

    r4, base4 = _delta_repo(
        tmp_path, tag="seedrt", base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '\n'                                                       # 2
                 'def test_the_round_trip_that_cannot_fail():\n'             # 3
                 '    assert parse(PAYLOAD) == PAYLOAD\n')                   # 4
    found = _mirrors(RV.honesty_prechecks(r4, base4, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 4, found

    r5, base5 = _seed_repo("seedexpr", tmp_path, 'target.write_text(PAYLOAD + "!")')
    found = _mirrors(RV.honesty_prechecks(r5, base5, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 7, found


@pytest.mark.parametrize("header", [
    "def helper(n=PAYLOAD):",
    "def helper(tmp_path, *, entity=PAYLOAD):",
    "def helper(\n    *, entity=PAYLOAD,\n):",
    "async def helper(\n    tmp_path,\n    entity=PAYLOAD,\n):",
])
def test_a_def_parameter_default_buys_the_constant_no_exclusion(header):
    """#1976: `_CALL_OPEN_RX` read a def's parameter list as a call's argument
    list, so a default made the constant a "fixture input" with no call in the
    file handing it anywhere — and silenced the identity round-trip in
    `tests/test_counterfactual_eval.py`."""
    src = f'PAYLOAD = "seed"\n{header}\n    ...\nassert read_back() == PAYLOAD\n'
    code = RV._code_only(src)[0]
    assert RV._constants_handed_in(code) == set()
    found = RV._mirrored_assertions(code)
    assert len(found) == 1 and found[0][2] == "PAYLOAD", found
    # The identity round-trip, the corpus shape itself.
    rt = (f'PAYLOAD = "seed"\n{header}\n    ...\n'
          'def test_x():\n    assert parse(PAYLOAD) == PAYLOAD\n')
    assert len(RV._mirrored_assertions(RV._code_only(rt)[0])) == 1


def test_the_def_skip_leaves_real_calls_and_lambdas_as_they_were():
    """The exclusion still excludes: a call is a call, in a def body or as a
    def's own default value."""
    for seed in ("target.write_text(PAYLOAD)", "target.write(content=PAYLOAD)"):
        src = (f'PAYLOAD = "seed"\ndef _seed(target):\n    {seed}\n'
               'def test_x():\n    assert read_back() == PAYLOAD\n')
        assert RV._constants_handed_in(RV._code_only(src)[0]) == {"PAYLOAD"}, seed
        assert RV._mirrored_assertions(RV._code_only(src)[0]) == [], seed
    nested = 'PAYLOAD = "seed"\ndef helper(n=wrap(PAYLOAD)):\n    ...\n'
    assert RV._constants_handed_in(RV._code_only(nested)[0]) == {"PAYLOAD"}
    lam = 'PAYLOAD = "seed"\nH = {"k": lambda n=PAYLOAD: n}\nassert read_back() == PAYLOAD\n'
    assert RV._constants_handed_in(RV._code_only(lam)[0]) == set()
    assert len(RV._mirrored_assertions(RV._code_only(lam)[0])) == 1


def test_prose_that_spells_a_write_buys_the_constant_no_exclusion(tmp_path):
    """#1864 clause 3: the seed scan reads `_code_only`'s blanked text, so a
    comment, a string literal and a module docstring all spelling
    `write_text(PAYLOAD)` are not the file putting a value into the world.

    Without that, a round could silence any mirror it was blamed for by writing
    one line of comment above the assertion — the exclusion would be a phrase, not
    a property of the code. The control is the same file with that call as real
    code, which is what makes the silence here about the prose and not about a
    detector that stopped running."""
    r, base = _delta_repo(
        tmp_path, tag="prosewrite", base_src=_MIRROR_BASE,
        post_src='PAYLOAD = "seeded identity"\n'                            # 1
                 '# the fixture used to write it: target.write_text(PAYLOAD)\n'   # 2
                 'HINT = "target.write_text(PAYLOAD)"\n'                     # 3
                 "'''A docstring naming target.write_text(PAYLOAD) as if it were"
                 " code.'''\n"                                               # 4
                 '\n'                                                        # 5
                 'def test_the_mirror_is_still_a_mirror():\n'                # 6
                 '    assert read_back() == PAYLOAD\n')                      # 7
    found = _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"],
                                          n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 7 and "PAYLOAD" in found[0]["problem"], found

    r2, base2 = _seed_repo("prosectl", tmp_path, "target.write_text(PAYLOAD)")
    assert _mirrors(RV.honesty_prechecks(r2, base2, ["tests/test_a.py"],
                                         n_clauses=1)) == []


def test_the_severity_ruling_is_written_where_the_constant_lives():
    """The comment above `_CONSTANT_MIRROR_SEVERITY` must carry BOTH severity
    rulings, and none of the pointers that used to stand in for them.

    The first ruling (measured 2026-09-29: 1/3 precision, 3 firings across 78
    graded rounds, both misses the fixture-seed class, over a window of one day)
    was itself only half an answer, because the block ended by pointing the reader
    at a longer window that had not been taken yet. That window has now been taken
    — measured again on 2026-10-07, 308 graded rounds and 11 firing rounds, of
    which 2 of the 11 were the input-and-expectation tautology and the other 9
    legitimate expectation-side pins — and promotion to `blocking` was refused on
    it too. So the block owes three things now: the numbers of both measurements,
    the name of the constant class the residual noise belongs to, and the removal
    of the pointer that routed a reader to an item's owed list instead of to the
    answer.

    A prose pin, deliberately, because the artefact under test IS prose: it is
    the sentence the next reader consults about whether to promote the severity,
    and a stale deferral there invites a re-measure of a question already closed.
    The severity itself is pinned as a value, not as text, by
    `tests/test_review_policy.py`."""
    src = (ROOT / "scripts" / "automod" / "review.py").read_text()
    block = (src.split("# ── the constant-mirroring assertion", 1)[1]
                .split("_CONSTANT_MIRROR_SEVERITY", 1)[0])
    assert "week of real rounds" not in block, block
    assert "settled by" not in block, "the block still defers the decision"
    for fact in ("2026-09-29", "1/3", "3 firings", "78 graded rounds",
                 "fixture seed"):
        assert fact in block, (fact, block)
    assert "one day" in block.lower(), "the window it actually measured is named"
    assert '_CONSTANT_MIRROR_SEVERITY = "advisory"' in src
    # Clause 1: the closed long-window measurement, named beside the one-day
    # baseline rather than in place of it — the second window confirmed the first,
    # so a block that kept only the new numbers would hide which is which.
    for fact in ("2026-10-07", "7-day", "308 graded rounds", "11 firing rounds",
                 "2 of the 11", "tautology"):
        assert fact in block, (fact, block)
    # Clause 2: nothing left that sends a reader somewhere else for the answer.
    assert "#1864" not in block, (
        "the item that pointer named is status: done with its owed entry settled; "
        "a dead pointer reads as a question still open")
    assert "still waiting" not in block, block
    assert "re-measur" not in block.lower(), (
        "a sentence about a measurement still to come is the deferral this block "
        f"has now retired twice: {block}")
    # Clause 3: the class the noise belongs to, and that a future look targets the
    # class and not the promotion decision.
    assert "expectation-side" in block, block
    for member in ("bound", "count", "golden digest",
                   "hand-written expected output"):
        assert member in block, (member, block)
    assert "not of promotion" in block, block


def test_a_constant_mirror_the_base_already_carried_is_not_blamed(tmp_path):
    """Clause 3: the delta is the same arithmetic as the five patterns. A base
    that already carried one mirrored assertion is blamed for none, and one
    that adds a second is blamed for exactly that one — at the new line."""
    carried = ("EXPECTED_TITLE = 'widget'\n"
               "\n"
               "def test_title_is_the_module_default():\n"
               "    assert parse(open_body()) == EXPECTED_TITLE\n")
    r, base = _delta_repo(tmp_path, tag="same", base_src=carried, post_src=carried)
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1)) == []
    # Same assertion twice over: still nothing new this round blamed for.
    r, base = _delta_repo(tmp_path, tag="twice", base_src=carried * 2, post_src=carried * 2)
    assert _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1)) == []
    # One MORE mirrored assertion is exactly one finding, naming the new node.
    r, base = _delta_repo(
        tmp_path, tag="more", base_src=carried,
        post_src=carried + "\nMAX_RETRIES = 3\n"                  # 6
                         "\n"                                     # 7
                         "def test_retry_default():\n"            # 8
                         "    assert load_config().retries == MAX_RETRIES\n")   # 9
    found = _mirrors(RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=1))
    assert len(found) == 1, found
    assert found[0]["line"] == 9, "the increase is the finding, not the carried one"
    assert "MAX_RETRIES" in found[0]["problem"] and "line 6" in found[0]["problem"]


def test_the_five_literal_patterns_still_refuse_after_the_new_detector(tmp_path):
    """Clause 5, first half: widening the check may not soften the five. The
    fixture is one round's worth of every one of them added to a file that had
    none, and each is still reported, once, at the severity its own table entry
    carries — `blocking`, which the last assertion here pins rather than trusts.
    The table is read rather than quoted so a softened or swapped entry fails
    this node instead of quietly agreeing with it."""
    expected = (r"\bor\s+True\b", r"^\s*assert\s+True\b", r"pytest\.skip\(",
                r"pytest\.mark\.skip", r"pytest\.mark\.xfail")
    assert tuple(p for p, _w, _s in RV._HONESTY_PATTERNS) == expected
    r, base = _delta_repo(tmp_path, base_src=_MIRROR_BASE,
                          post_src=_fixture("five_literal_patterns.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py", "app/m.py"], n_clauses=2)
    for _pat, why, severity in RV._HONESTY_PATTERNS:
        hit = [o for o in out if o["problem"] == why]
        assert len(hit) == 1, (why, out)
        assert hit[0]["severity"] == severity == "blocking", (why, out)


def test_a_pattern_spelled_in_a_string_is_not_counted_as_new_dishonest_code(tmp_path):
    """The prechecks run on the round's OWN changed test files, so a round that
    adds a test for this checker has to put the five shapes inside the source
    strings of that test. Prose about a pattern is not the pattern — and the
    control at the end is the same file with one of them written as code, which
    still refuses, on the code's own line. This is the shape that refused
    `SM_20260928_210355` for adding the tests that pin it."""
    prose = _fixture("patterns_named_in_prose_base.txt")
    r, base = _delta_repo(
        tmp_path, tag="prose", base_src=prose,
        post_src=prose + _fixture("patterns_named_in_prose_added.txt"))
    assert RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0) == []
    r, base = _delta_repo(
        tmp_path, tag="code", base_src=prose,
        post_src=prose + _fixture("real_assert_true_added.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    hits = [o for o in out if o["problem"] == _why(r"assert\s+True")]
    assert len(hits) == 1, out
    assert hits[0]["severity"] == "blocking" and hits[0]["line"] == 8, out


def test_a_pattern_in_a_docstring_is_prose_and_the_line_reported_is_the_code_s(tmp_path):
    """The blank pass has to be blind across MULTIPLE lines, not just across a
    quoted word, and it may not move the lines it reports: a three-line
    docstring that names a skip call and a bare-true assertion is prose, and the
    one real assertion below it is on line 6 however many lines the docstring
    swallowed."""
    r, base = _delta_repo(
        tmp_path, tag="doc", base_src=_MIRROR_BASE,
        post_src=_fixture("multiline_docstring_then_real_assert.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    assert len(out) == 1, out
    assert out[0]["problem"] == _why(r"assert\s+True"), out
    assert out[0]["line"] == 6, "the code's line, not the one the blanking left"


def test_a_conditional_skip_is_still_demoted_to_advisory(tmp_path):
    """Clause 5, second half — the #1204 fix. 27 of 134 refusals were skip-only
    with every clause graded met; a skip behind a condition is a judgment and
    goes to the grader as advice, not as a refusal."""
    r, base = _delta_repo(tmp_path, base_src=_MIRROR_BASE,
                          post_src=_fixture("conditional_skip.txt"))
    out = RV.honesty_prechecks(r, base, ["tests/test_a.py"], n_clauses=0)
    skips = [o for o in out if o["problem"].startswith(_why(r"skip\("))]
    assert len(skips) == 1, out
    assert skips[0]["severity"] == "advisory", out
    assert "conditional" in skips[0]["problem"], out


def test_the_new_detector_scans_the_files_testpaths_owns(tmp_path):
    """Clause 5, third half: the file→test mapping is `TP.pick_test_files`'s
    answer read from `pytest.ini` — the same one the patterns and #1322's
    met-node rail read — not a `tests/` prefix and no new resolver. A mirrored
    assertion in a harness test under a second testpath is reported; one in
    ordinary code under no testpath is not."""
    mirrored = "EXPECTED_TITLE = 'widget'\n" \
               "\n" \
               "def test_title_is_the_module_default():\n" \
               "    assert parse(open_body()) == EXPECTED_TITLE\n"
    r, base = _delta_repo(
        tmp_path, base_src=_MIRROR_BASE,
        post_src=_MIRROR_BASE + "\n" + mirrored,
        pytest_ini="[pytest]\ntestpaths = tests app/harness/tests scripts\n",
        new_files={"app/harness/tests/test_h.py": mirrored,
                   "app/code.py": mirrored})
    out = RV.honesty_prechecks(r, base,
                               ["tests/test_a.py", "app/harness/tests/test_h.py",
                                "app/code.py", "app/m.py"], n_clauses=1)
    files = sorted({o["file"] for o in _mirrors(out)})
    assert files == ["app/harness/tests/test_h.py", "tests/test_a.py"], out
    assert [p for p in RV.TP.pick_test_files(
        ["tests/test_a.py", "app/harness/tests/test_h.py", "app/code.py"], r)] == \
        ["tests/test_a.py", "app/harness/tests/test_h.py"]


# ── every citation is validated, whatever verdict carries it (#1442) ───────
# Round SM_20260924_104307, review attempt 2 of 2: four `partial`s refused the
# round, and every checkable claim in them was false of the head the gate
# named. The evidence was a test file that exists in no commit
# (`tests/test_facts_surviving_readers.py`) and a landing at `08a4f4f0`, which
# `git cat-file -t` calls `fatal: Not a valid object name` in both `~/lloyd` and
# the vault. Both rails existed and both were skipped: `_node_rail`'s existence
# check and `normalize_evidence_path`'s answer were consulted only inside
# `if verdict == "met":`, so a uniformly-`partial` restatement — the shape the
# finalizer's schema pass emits, not the grading turn, which returned
# `verdict: approve` — was checked against nothing. `unresolved_shas` and
# `added_test_denials` carry the mechanism.

PHANTOM = "tests/test_facts_surviving_readers.py::test_the_surviving_kg_readers_still_register"


def _fin(*specs) -> dict:
    """A `clauses` list in the shape the finalizer emits: one entry per clause."""
    clauses = []
    for i, spec in enumerate(specs, 1):
        c = {"clause": i, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 3,
             "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran", "note": "ok"}
        c.update(spec)
        clauses.append(c)
    return {"premise": "sound", "clauses": clauses, "test_honesty": [],
            "seams_unverified": [], "summary": "fine"}


def _judged(wt, obj, n, **kw):
    return RV.parse_review(obj, worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=n,
                           tests_passed=True, changed_paths=["app/x.py", "tests/test_x.py"], **kw)


@pytest.mark.parametrize("verdict", ["met", "partial", "unmet"])
def test_a_test_node_absent_from_the_tree_is_rejected_on_every_verdict(wt, verdict):
    """`partial` is precisely the verdict that skipped the check on 2026-09-24,
    so the rail cannot live only in the `met` branch."""
    parsed = _judged(wt, _fin({"verdict": verdict, "test_node_id": PHANTOM}), 1)
    c = parsed["clauses"][0]
    assert c.get("citation_unresolved"), "an entry naming an absent test file is marked whatever its verdict"
    assert "tests/test_facts_surviving_readers.py" in c["citation_unresolved"][0]
    assert parsed["unreliable"], "a refusal built on a file that is not in the graded tree is not a verdict"


@pytest.mark.parametrize("verdict", ["met", "partial", "unmet"])
def test_an_evidence_path_that_resolves_to_nothing_is_recorded_on_the_clause(wt, verdict):
    """The refusal then says the citation failed instead of silently
    accepting `agent_mcp/facts.py:520-540`-shaped text for a test. An evidence
    path alone stays a graded refusal, not an unusable review."""
    parsed = _judged(wt, _fin({"verdict": verdict, "evidence_path": "app/facts_gone.py:520-540"}), 1)
    c = parsed["clauses"][0]
    assert c.get("citation_unresolved") and "app/facts_gone.py" in c["citation_unresolved"][0]
    assert parsed["unreliable"] == [], "a path the grader mis-cited is still a finding about the diff"


def test_a_review_whose_entries_all_cite_absent_files_is_unusable(wt):
    """The 2026-09-24 refusal: all four entries named a file in no tree, branch
    or commit, and the round still spent its last attempt on the text."""
    specs = [{"verdict": "partial", "test_node_id": f"tests/test_ghost_{i}.py::test_x"}
             for i in range(4)]
    parsed = _judged(wt, _fin(*specs), 4)
    assert len(parsed["clauses"]) == 4
    assert all(c.get("citation_unresolved") for c in parsed["clauses"])
    assert any("every" in r.lower() for r in parsed["unreliable"]), parsed["unreliable"]


def test_a_commit_cited_in_a_note_that_is_not_an_object_makes_the_review_unreliable(repo):
    """`git cat-file -t 08a4f4f0` → fatal, yet the rung wrote "a test in a prior
    landing … commit 08a4f4f0 … that I ran and read" and the gate believed it."""
    r, _ = repo
    head = git(r, "rev-parse", "HEAD").stdout.strip()
    bad = _judged(r, _fin({"verdict": "partial", "test_node_id": "",
                           "evidence_path": "app/m.py", "evidence_line": 1,
                           "note": "the pin exists from commit 08a4f4f0, not in this diff's tests"}),
                  1, repo=r)
    assert bad["clauses"][0]["citation_unresolved"]
    assert "08a4f4f0" in bad["clauses"][0]["citation_unresolved"][0]
    assert bad["unreliable"] and "08a4f4f0" in " ".join(bad["unreliable"])
    # Positive control through the same rail: a sha the repo really has.
    good = _judged(r, _fin({"verdict": "partial", "test_node_id": "", "evidence_path": "app/m.py", "evidence_line": 1,
                            "note": f"the pin landed at {head}, and the suite is green"}), 1, repo=r)
    assert good["unreliable"] == [] and "citation_unresolved" not in good["clauses"][0]


def test_a_repo_the_rail_cannot_read_invents_no_unresolved_shas(wt):
    """Fourth instance of the catalogued class: a guard that reads its own
    missing input reports what it cannot see. `tmp_path` is not a repo, so
    `git cat-file` answers nothing, and the answer is to stay quiet."""
    parsed = _judged(wt, _fin({"verdict": "partial", "test_node_id": "",
                              "note": "satisfied by commit 08a4f4f0"}), 1, repo=wt)
    assert parsed["unreliable"] == [] and "citation_unresolved" not in parsed["clauses"][0]


def test_the_sha_rail_asks_git_and_does_not_read_prose_as_a_commit(repo):
    """The shape filter is a cheap pre-check; the git lookup is the whole
    verdict. Hex-looking English (`decode`, `beadded`), an all-decimal date
    (`20260915`) and a short line range are not commit-ish at all, while a
    7-token with a digit and a letter IS asked — and a real sha in the same
    note is the positive control that the asking works."""
    r, _ = repo
    head = git(r, "rev-parse", "HEAD").stdout.strip()
    quiet = _judged(r, _fin({"verdict": "partial", "test_node_id": "", "evidence_path": "app/m.py", "evidence_line": 1,
                             "note": f"the decode path, the beadded case, since 20260915, "
                                     f"agent_mcp/facts.py:520-540 and {head[:12]} are all "
                                     f"consistent"}), 1, repo=r)
    assert quiet["unreliable"] == [] and "citation_unresolved" not in quiet["clauses"][0]
    # The same note with one token the repo does not have.
    # `0f`×6 keeps the shape filter (hex, 12 chars, has both a digit and a
    # letter) and is an object no repo has.
    loud = _judged(r, _fin({"verdict": "partial", "test_node_id": "", "evidence_path": "app/m.py", "evidence_line": 1,
                            "note": quiet["clauses"][0]["note"].replace(head[:12], "0f" * 6)}),
                   1, repo=r)
    assert "0f0f0f0f0f0f" in " ".join(loud["unreliable"]), "the lookup decides, the shape only asks"


def test_the_note_of_an_unsound_premise_is_marked_but_its_verdict_still_stands(wt):
    """The unsound call is the grader's to make about the ITEM, and an
    unusable-citation flag must not swallow it."""
    obj = _fin({"verdict": "partial", "test_node_id": PHANTOM})
    obj["premise"] = "unsound"
    parsed = _judged(wt, obj, 1)
    assert parsed["premise"] == "unsound" and parsed["unreliable"]


# ── the rung: four outcomes, and the flags ride the event ──────────────────

class _Gate(G.Gate):
    def __init__(self, item_id, changed, tmp_path):
        super().__init__("SM_REV", tmp_path, "a" * 40, item_id=item_id)
        self.report.changed_paths = changed
        self.report.rungs.append(G.RungResult("tests", True, "ok", 1.0, {"passed": 10}))


def _arm(monkeypatch, tmp_path, *, grade, contract=None, prior=0,
         prior_head: str = "", prior_clauses=None):
    """`prior_head` makes every recorded refusal a refusal of ONE commit, which
    is what the answer-from-the-ledger branch keys on; `prior_clauses` is what
    that recorded refusal graded, and the rows a replayed refusal has to quote
    with its own findings (#2448). Both default to what they were: distinct
    heads, and a refusal the ledger never gave verdicts for."""
    events: list[dict] = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    # Graded refusals of DISTINCT commits: only those spend an attempt now.
    monkeypatch.setattr(G.S, "read_events", lambda limit=100: [
        {"event": "review", "round_id": "SM_REV", "ok": True, "blocking": True,
         "attempt": i + 1, "head": prior_head or f"{i:040d}", "findings": f"f{i}",
         **({"clauses": list(prior_clauses)} if prior_clauses is not None else {})}
        for i in range(prior)])
    monkeypatch.setattr(G.W, "round_dir", lambda rid: tmp_path / "round")
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: contract or {
        "id": iid, "title": "t", "body": "b", "clauses": ["the thing happens once"], "path": ""})
    monkeypatch.setattr(RV, "grade", grade)
    monkeypatch.setattr(RV, "honesty_prechecks", lambda *a, **k: [])
    return events


def _grader(structured, *, ok=True, error=""):
    def grade(**kw):
        grade.calls.append(kw)
        return {"ok": ok, "error": error, "session_id": "sess_r", "structured": structured,
                "structured_error": "", "text": "", "stop_reason": "stop", "duration_s": 1.0}
    grade.calls = []
    return grade


def test_no_item_bound_is_a_recorded_skip(monkeypatch, tmp_path):
    """A human's round has no contract. That is a skip that says so, not a
    pass that pretends to have graded."""
    _arm(monkeypatch, tmp_path, grade=_grader(None))
    g = _Gate(None, ["app/x.py"], tmp_path)
    ok, detail, data = g.rung_review()
    assert ok and data["skipped"] and "no backlog item" in detail


def test_an_unreachable_grader_is_external_not_a_pass(monkeypatch, tmp_path):
    """A waived review is the #544 shape. The engine being down is the
    engine's fault, so the item keeps its attempt — but the rung fails."""
    events = _arm(monkeypatch, tmp_path, grade=_grader(None, ok=False, error="HTTP 503"))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True and "503" in detail
    assert events[-1]["event"] == "review" and events[-1]["ok"] is False


def _phantom_entry(verdict="partial", node=PHANTOM, note="the pin is elsewhere"):
    return {"clause": 1, "verdict": verdict, "evidence_path": "app/x.py", "evidence_line": 1,
            "test_node_id": node, "how_verified": "ran", "note": note}


def _one_tree(tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_it():\n    assert 1\n")


def test_an_unresolvable_citation_makes_the_review_unreliable_and_spends_no_attempt(
        monkeypatch, tmp_path):
    """The 2026-09-24 round died on this text. A grader whose own evidence is
    not in the tree it was handed has not judged the diff, so the item keeps its
    attempt exactly as it does when the grader is unreachable."""
    _one_tree(tmp_path)
    obj = {"premise": "sound", "summary": "close", "clauses": [_phantom_entry()],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), prior=1)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True
    assert "unreliable" in detail and "keeps its attempt" in detail
    assert "review_retry" not in data, "an unreliable review is not a graded refusal"
    assert "tests/test_facts_surviving_readers.py" in detail
    assert events[-1]["ok"] is False and events[-1]["blocking"] is False
    assert "test_facts_surviving_readers" in events[-1]["error"]


# The two wordings this rail has, verbatim from review.py. `PAST_EOF` is the
# `downgraded` reason #1750 clause 5 holds to, and the one that reaches the review event
# on the re-ask path; `PAST_EOF_UNRESOLVED` is the `citation_unresolved` wording, which
# is what a broken citation rail carries into `unreliable` (#1442's shape).
PAST_EOF = "evidence_line {line} past EOF ({eof} lines) of {path}"
PAST_EOF_UNRESOLVED = "evidence_line {line} is past EOF of {path} ({eof} lines at the graded head)"


def _past_eof_entry(clause=1, *, line=500, **kw):
    """A `met` clause in a `_one_tree` worktree: `app/x.py` is one line there, so 500
    is past EOF and every other rail holds. Each keyword is a rail #1750 says must
    hold before a citation defect may be treated as the grader's own problem.
    """
    entry = {"clause": clause, "verdict": "met", "evidence_path": "app/x.py",
             "evidence_line": line, "test_node_id": "tests/test_x.py::test_it",
             "how_verified": "ran",
             "note": "capture_round plants each scenario and the scorecard names the dir"}
    entry.update(kw)
    return entry


def test_a_met_clause_whose_only_defect_is_a_line_past_eof_is_not_a_grade(wt):
    """Clause 1: the only rail that fails is a line the named file cannot contain, so
    `parse_review` reports the review unreliable rather than returning an empty list.

    The distinction decides what the attempt is spent on. A wrong line number in a
    file that resolved is arithmetic the grader did, and the author has nothing to
    edit — the file, the node and the behaviour all passed. `unreliable` is the list
    that means "this review did not grade the diff", so adding an entry here IS the
    fix: the no-attempt path for that list already exists in `rung_review`.
    """
    parsed = RV.parse_review(_obj(evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is True, c
    assert c["downgraded"] == [PAST_EOF.format(line=500, eof=6, path="app/x.py")], \
        "the reason stays verbatim in the clause however the review is routed"
    assert len(parsed["unreliable"]) == 1, parsed["unreliable"]
    assert "clause 1" in parsed["unreliable"][0]
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in parsed["unreliable"][0]


def test_a_line_past_eof_spends_no_attempt_at_the_rung(monkeypatch, tmp_path):
    """Clause 2: at the rung that same review is an `external_blocker`, not a retry.

    Shaped like `test_an_unresolvable_citation_makes_the_review_unreliable_and_spends_no_attempt`,
    which pins this accounting for a test file the grader invented. The line-number
    case reaches the same arm and inherits the same treatment: the rung fails, the item
    keeps its attempt, and `review_retry` is absent because nothing was graded.
    """
    _one_tree(tmp_path)
    obj = {"premise": "sound", "summary": "graded, but on a number that is not there",
           "clauses": [_past_eof_entry()], "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), prior=1)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True, data
    assert "keeps its attempt" in detail, detail
    assert "no review attempt is spent" in detail, detail
    assert "review_retry" not in data, "a grader's arithmetic is not a graded refusal"
    assert PAST_EOF.format(line=500, eof=1, path="app/x.py") in detail, detail
    ev = events[-1]
    assert ev["event"] == "review" and ev["ok"] is False and ev["blocking"] is False
    assert PAST_EOF.format(line=500, eof=1, path="app/x.py") in ev["error"], ev["error"]


@pytest.mark.parametrize("bad_node", [
    # A real node, in a test file this diff never touched: the clause is demoted, and
    # the author can fix it by citing a test the diff actually changed.
    "tests/test_old.py::test_before",
    # A file the grader invented. This is the rail that already routes to `unreliable`
    # on its own (#1442), and the case the flag must not swallow: the phantom-file
    # `broken` marker is the ONLY thing separating this from a re-ask, since `why`
    # still holds nothing but the past-EOF line.
    PHANTOM,
])
def test_a_line_past_eof_beside_a_failing_node_rail_is_still_a_graded_refusal(wt,
                                                                             bad_node):
    """Clause 3: past EOF is the grader's own problem only when it is the ONLY one.

    With the node rail failing as well, the clause has a defect the author can act on
    (or the review is already unreliable for a reason that has its own test), so the
    citation gets no special treatment and `unreliable` stays as those tests left it.
    Widening the routing past this line would hand every bad node citation a free
    re-gate, and a grader that keeps inventing nodes would never be refused at all.
    """
    parsed = RV.parse_review(
        _obj(evidence_line=500, test_node_id=bad_node),
        worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] in ("partial", "unmet"), c
    # The flag is what routes a clause to the re-ask, and it is not set for either
    # shape: a second rail spoke about the clause, whatever it decided.
    assert c.get("citation_only") is not True, c
    if bad_node == "tests/test_old.py::test_before":
        # A node in a file the diff never touched is a finding about the diff, so the
        # review stays actionable: `unreliable` empty, no free re-gate, the attempt
        # charged as before. And the demoted clause still carries the reason verbatim.
        assert parsed.get("unreliable", []) == [], parsed.get("unreliable")
        assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in c["downgraded"], c
    else:
        # The invented-file shape was ALREADY unreliable before this change (#1442):
        # the rail could not read its input at all, so `broken` named the clause and the
        # review spent no attempt. That path is unchanged; the past-EOF line rides along
        # inside the SAME reason list, because `broken[idx]` holds the very list the line
        # rail appends to. Both reasons are visible, neither one is a new route.
        assert c["verdict"] == "partial", c
        assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in c["downgraded"], c
        assert any("test_node_id" in w for w in c["downgraded"]), c
        reasons = " ".join(parsed["unreliable"])
        assert bad_node.split("::")[0] in reasons, parsed["unreliable"]
        assert PAST_EOF_UNRESOLVED.format(line=500, eof=6, path="app/x.py") in reasons, \
            parsed["unreliable"]
        reasons = parsed["unreliable"]
        assert any(bad_node.split("::")[0] in r for r in reasons), reasons
        assert PAST_EOF_UNRESOLVED.format(line=500, eof=6, path="app/x.py") in \
            " ".join(reasons), reasons


def test_a_line_past_eof_beside_an_inferred_verification_is_a_graded_refusal(wt):
    """Clause 3, other half: `how_verified: inferred` is a claim about the evidence,
    not about the grader's arithmetic, so the clause stays a `partial` to be fixed.
    """
    parsed = RV.parse_review(_obj(evidence_line=500, how_verified="inferred"),
                             worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is not True, c
    assert parsed.get("unreliable", []) == [], parsed.get("unreliable")


def test_a_demotion_that_is_not_a_phantom_line_is_still_a_grade(wt):
    """The guard's other half: `len(why) == 1` alone would be enough to re-ask almost
    every demotion in the file, so the reason has to BE the impossible line.

    A single `how_verified is not ran|read` demotion is a finding about the clause: the
    grader looked and could not say it ran anything. Granting that a re-ask would let a
    round re-gate indefinitely without the author changing a line, and would empty
    `unreliable` of its meaning — the list says the review could not check the diff, not
    that the diff was weak.
    """
    parsed = RV.parse_review(_obj(how_verified="inferred"), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c["downgraded"] == ["how_verified is not ran|read"], c
    assert c.get("citation_only") is not True, c
    assert parsed.get("unreliable", []) == [], parsed.get("unreliable")
    assert parsed["downgraded"] == [1], "the downgrade is still a graded fact"


def test_a_phantom_line_on_a_broken_premise_still_spends_its_attempt(monkeypatch,
                                                                    tmp_path):
    """A citation defect must not rescue a review that also says the item is unsound.

    `PREMISES` is exactly `("sound", "unsound")`, and the no-attempt arm returns the text
    `the item keeps its attempt`; reaching it from an unsound premise would hand a round
    the grader says should not have been attempted a free re-gate every time the grader
    also mis-numbers a line. The premise verdict is a finding about the ITEM, so it
    outranks the citation and the refusal charges its attempt as it did before.
    """
    _one_tree(tmp_path)
    obj = {"premise": "unsound", "premise_problems": ["the item restates a landed fix"],
           "summary": "", "clauses": [_past_eof_entry()],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), prior=1)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data.get("external_blocker") is not True, data
    # `Gate.rung_review`'s unsound-premise branch returns `review_premise_unsound`, not
    # `review_retry`, and it is a charged refusal — so the key is the branch, named.
    assert data["review_premise_unsound"] is True, data
    assert "keeps its attempt" not in detail, detail
    # Charged, in the rung's own ledger event: `blocking` true and the unsound-premise
    # `kind`, which is a different decision from the graded `retry` — a finding about the
    # item, not about the diff.
    ev = events[-1]
    assert ev["blocking"] is True and ev["kind"] == "unsound", ev
    # The item-level keys `review_retry` / `review_premise_unsound` are copied onto the
    # event by the gate's run loop (`gate.py`'s
    # `for key in ("review_retry", "review_premise_unsound")`), outside this rung, so
    # they are asserted in `data`, which is what that loop reads.
    assert data["review_summary"], data


def test_a_line_past_eof_never_masks_an_unmet_clause(monkeypatch, tmp_path):
    """Clause 4: one phantom line number must not hide a genuine finding on another
    clause, because the `unreliable` arm returns before findings are ever delivered.

    This is the common shape, not the corner: the docstring above `parse_review`
    counts 23 `met` clauses across 15 rounds citing a line past EOF. A round that
    pairs a phantom number with a real failure still has to come back with that
    failure named, and still pays for it.
    """
    _one_tree(tmp_path)
    contract = {"id": 7, "title": "t", "body": "b", "path": "",
                "clauses": ["clause one holds", "clause two holds"]}
    obj = {"premise": "sound", "summary": "one phantom, one real", "clauses": [
        _past_eof_entry(1),
        {"clause": 2, "verdict": "unmet", "evidence_path": "app/x.py", "evidence_line": 1,
         "test_node_id": "tests/test_x.py::test_it", "how_verified": "ran",
         "note": "the second path is not covered at all"}],
        "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), contract=contract)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False, "a real unmet clause must still refuse the round"
    assert data.get("external_blocker") is not True, (
        "the phantom citation swallowed the finding: the author would be told to "
        "re-gate and never shown what is wrong")
    assert data["review_retry"] is True and data["review_attempt"] == 1, data
    assert "the second path is not covered at all" in data["review_findings"], data
    assert PAST_EOF.format(line=500, eof=1, path="app/x.py") in data["review_findings"], \
        "the phantom line is still reported, as a downgrade rather than a re-ask"
    # Charged, and both clause texts reach the ledger: `blocking` + the graded `retry`
    # kind, `findings` naming the real failure, and the phantom clause still carrying its
    # verbatim `downgraded` reason and its `citation_only` mark.
    ev = events[-1]
    assert ev["blocking"] is True and ev["kind"] == "retry", ev
    assert "the second path is not covered at all" in ev["findings"], ev
    phantom = next(c for c in ev["clauses"] if c["clause"] == 1)
    assert phantom["citation_only"] is True and phantom["downgraded"] == [
        PAST_EOF.format(line=500, eof=1, path="app/x.py")], phantom
    assert ev["downgraded"] == [1], ev


def test_the_reread_still_loses_nothing_in_the_ledger(monkeypatch, tmp_path):
    """Clause 5, the half that is not about the verdict: the re-read leaves the reason
    in the ledger word for word, in BOTH places the review event carries it.

    Three rounds on 2026-09-11 were downgraded here and nothing recorded which path
    failed. Converting the case into a re-ask must not lose the reason along with the
    refusal. The one `review` event holds the string twice over: in `error`, the
    human-readable reason the arm returns, and in `clauses[0]["downgraded"]`, the
    machine-readable list the finalizer reads. A reader searching `promotions.jsonl`
    for `evidence_line` finds it either way.

    The refusal path is covered by the parametrised
    `test_a_line_past_eof_beside_a_failing_node_rail_is_still_a_graded_refusal`, which
    asserts the same string in that clause's `downgraded`, and by
    `test_a_line_past_eof_never_masks_an_unmet_clause`, which asserts it inside the
    event's `findings`.
    """
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")
    obj = {"premise": "sound", "summary": "", "clauses": [_past_eof_entry()],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    review_events = [e for e in events if e["event"] == "review"]
    assert len(review_events) == 1, events
    ev = review_events[0]
    assert reason in ev["error"], ev
    assert ev["clauses"][0]["downgraded"] == [reason], ev["clauses"][0]
    assert ev["clauses"][0]["citation_only"] is True, ev["clauses"][0]
    # The list the no-attempt arm is built on, on the event too: this is what
    # `rung_review`'s detail and the item's attempt accounting both read.
    assert any(reason in u for u in ev["unreliable"]), ev.get("unreliable")


# ── #1845: an advisory test-honesty note may not cost the round its re-ask ────
#
# Round SM_20260929_101355 (#1812, 2026-09-29T10:35:13Z) is the shape these tests
# hold: clause 1 demoted for a line past EOF and marked `citation_only`, clauses
# 2-5 `met`, premise `sound`, and two `severity: advisory` test-honesty notes.
# The block-list beside the re-ask read a `clause` key that no honesty entry has
# ever carried, so it counted those two notes as findings the author could act
# on, `unreliable` stayed empty, and the round paid for a review that had graded
# nothing (attempt 1's 220.7 s plus a 206.3 s test re-run before attempt 2 landed
# it). `decide_by_grader` had already called those same entries advisory.

def _honesty(severity="advisory", *, actionable=True,
             problem="the assert message restates the literal it compares"):
    """One grader test-honesty finding, filed against a test file this diff changed.

    The path matters as much as the severity: `parse_review` demotes any finding
    about a non-test path to `advisory` whatever the grader wrote, so naming
    `tests/test_x.py` is what makes a `severity: "blocking"` entry a test of the
    severity and not an accidental test of the file."""
    return {"file": "tests/test_x.py", "line": 2, "problem": problem,
            "severity": severity, "actionable_in_round": actionable}


def _phantom_plus_met_obj(*honesty):
    """Two clauses in a `_one_tree`/`wt` worktree: clause 1 cites line 500 of a file
    that cannot hold it, clause 2 is met on evidence inside the same file.

    Both fixtures serve: `app/x.py` is 6 lines in `wt` and 1 in `_one_tree`, so the
    phantom reason reads `(6 lines)` in one and `(1 lines)` in the other, and clause
    2's `evidence_line` 1 is inside either way. Clause 2 present and `met` is what
    makes this the live shape — the block-list asks about EVERY clause, not just the
    demoted one."""
    return {"premise": "sound",
            "summary": "clause 1 graded on a line the named file cannot hold",
            "clauses": [_past_eof_entry(1, note="the guard refuses and prints the flag"),
                        _past_eof_entry(2, line=1, note="the dry run still prints")],
            "test_honesty": list(honesty), "seams_unverified": []}


_TWO_CONTRACT = {"id": 7, "title": "t", "body": "b", "path": "",
                 "clauses": ["clause one holds", "clause two holds"]}


def test_an_advisory_honesty_note_does_not_cost_the_round_its_past_eof_reask(wt):
    """Clause 1: a note the file's own policy calls advisory leaves the re-ask intact.

    `unreliable` is the whole of the no-attempt route, so an entry appearing there IS
    the fix; the note is not lost, it rides on in `test_honesty` for the report.
    """
    parsed = RV.parse_review(_phantom_plus_met_obj(_honesty("advisory")), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=2)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is True, c
    assert len(parsed["unreliable"]) == 1, (
        f"the advisory note suppressed the re-ask: {parsed['unreliable']}")
    assert "clause 1" in parsed["unreliable"][0], parsed["unreliable"]
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in parsed["unreliable"][0], \
        parsed["unreliable"]
    assert parsed["test_honesty"][0]["severity"] == "advisory", (
        "the note must still reach the report, not be dropped to make the route work")
    # And the same entry is advisory on the decide side too. A review routed to the
    # re-ask never reaches `decide_by_grader` — the `unreliable` arm in `rung_review`
    # returns first — so this call, not the rung, is where the two surfaces can be
    # compared for this shape. The note may only ever append a tail to the identical
    # refusal the review makes without it.
    kind0, findings0 = RV.decide_by_grader(dict(parsed, test_honesty=[]), [])
    kind, findings = RV.decide_by_grader(parsed, [])
    assert kind == kind0 == "retry", (kind, kind0, findings)
    assert findings.startswith(findings0 + "; "), (findings0, findings)
    assert "advisory tests/test_x.py:2: " in findings[len(findings0):], findings


def test_a_blocking_note_the_round_cannot_fix_also_leaves_the_reask(wt):
    """Clause 2: `actionable_in_round: false` demotes too, exactly as
    `decide_by_grader` already reads it — asserted by calling that function on the
    same parsed review, so the two surfaces cannot drift back apart unnoticed."""
    parsed = RV.parse_review(_phantom_plus_met_obj(_honesty("blocking", actionable=False)),
                             worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=2)
    assert len(parsed["unreliable"]) == 1, (
        f"a blocking-but-unfixable note suppressed the re-ask: {parsed['unreliable']}")
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in parsed["unreliable"][0], \
        parsed["unreliable"]
    # `decide_by_grader` on the SAME parsed review is the policy this has to match.
    # The control is the same review with the note removed: the note may only ever
    # append an advisory tail to an identical refusal, never add a blocking line —
    # which is exactly the distinction the block-list was getting wrong.
    kind0, findings0 = RV.decide_by_grader(dict(parsed, test_honesty=[]), [])
    kind, findings = RV.decide_by_grader(parsed, [])
    assert kind == kind0 == "retry", (kind, kind0, findings)
    assert "clause 1 partial" in findings0, findings0
    assert findings.startswith(findings0 + "; "), (findings0, findings)
    tail = findings[len(findings0):]
    assert "advisory test honesty tests/test_x.py:2" in tail, tail
    assert "(blocking, but not fixable in this round)" in tail, tail


def test_a_blocking_actionable_note_still_refuses_instead_of_re_asking(wt):
    """Clause 3, both halves: a finding that is blocking AND fixable in this round
    buys no re-ask (`unreliable` empty), and the author is still shown it while the
    demoted clause is still refused as a partial."""
    parsed = RV.parse_review(_phantom_plus_met_obj(_honesty("blocking")), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=2)
    assert parsed.get("unreliable", []) == [], (
        f"a fixable honesty finding was routed to the free re-gate: {parsed['unreliable']}")
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is True, c
    assert PAST_EOF.format(line=500, eof=6, path="app/x.py") in c["downgraded"], c
    kind, findings = RV.decide_by_grader(parsed, [])
    kind0, findings0 = RV.decide_by_grader(dict(parsed, test_honesty=[]), [])
    assert kind == kind0 == "retry", (kind, kind0, findings)
    assert "test honesty" not in findings0, (
        "the line this test is about has to come from the honesty entry:\n" + findings0)
    assert "test honesty tests/test_x.py:2: " in findings, (
        "the finding the author can fix in this round was not shown to them:\n" + findings)
    assert "clause 1 partial" in findings, findings


def test_the_advisory_note_buys_the_reask_and_the_blocking_note_costs_an_attempt(
        monkeypatch, tmp_path):
    """Clause 4, both halves at the rung: the advisory shape is an `external_blocker`
    that spends nothing, the blocking-actionable shape is a graded refusal that spends
    attempt 1. One test because the pair is the whole of the change's value: routing
    everything to the no-attempt arm would be as wrong as charging both."""
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")

    events = _arm(monkeypatch, tmp_path, grade=_grader(_phantom_plus_met_obj(
        _honesty("advisory"))), contract=_TWO_CONTRACT)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True, data
    assert "keeps its attempt" in detail, detail
    assert "no review attempt is spent" in detail, detail
    assert "review_retry" not in data, (
        "an advisory note sent the round down the charged-refusal path:\n" + detail)
    assert reason in detail, detail
    # The no-attempt arm writes one `review` event and nothing else. Which arm ran
    # is not in that event — `self.event(...)` forwards only error/clauses/
    # seams_unverified/test_honesty/unreliable, so `external_blocker` and
    # `review_retry` above are the rung's `data`, not the ledger's — so what the
    # record can prove is that the reason survived on the no-attempt path.
    assert len(events) == 1, events
    assert any(reason in u for u in events[0].get("unreliable", [])), events[0]

    _arm(monkeypatch, tmp_path, grade=_grader(_phantom_plus_met_obj(
        _honesty("blocking"))), contract=_TWO_CONTRACT)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False, detail
    assert data.get("external_blocker") is not True, (
        "a finding the author can fix in this round was forgiven as the grader's "
        "own arithmetic:\n" + detail)
    assert data["review_retry"] is True and data["review_attempt"] == 1, data
    assert "test honesty tests/test_x.py:2" in data["review_findings"], data


def test_the_advisory_reask_still_puts_the_past_eof_reason_in_the_ledger(
        monkeypatch, tmp_path):
    """Clause 5, the record half: routing the case to the re-ask because of an
    advisory note must not lose the reason, which the event carries twice over — in
    `error` for a human and in `clauses[0]["downgraded"]` for the finalizer.

    Same two places `test_the_reread_still_loses_nothing_in_the_ledger` pins for a
    review with no honesty notes at all; this is the note-present half of that claim,
    and the half that was false while the block-list counted any note as blocking."""
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")
    events = _arm(monkeypatch, tmp_path, grade=_grader(_phantom_plus_met_obj(
        _honesty("advisory"))), contract=_TWO_CONTRACT)
    _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    review_events = [e for e in events if e["event"] == "review"]
    assert len(review_events) == 1, events
    ev = review_events[0]
    assert ev["ok"] is False and ev["blocking"] is False, ev
    assert reason in ev["error"], ev
    phantom = next(c for c in ev["clauses"] if c["clause"] == 1)
    assert phantom["citation_only"] is True and phantom["downgraded"] == [reason], phantom
    assert any(reason in u for u in ev["unreliable"]), ev.get("unreliable")


def test_an_advisory_note_beside_a_phantom_line_and_an_unmet_clause_is_a_charged_refusal(
        monkeypatch, tmp_path):
    """Clause 5, the refusal half: the re-ask stays conditional on the phantom line
    being the review's ONLY defect, honesty notes included.

    One phantom number, one genuinely unmet clause, one advisory note. The rung
    short-circuits on `unreliable` before findings are delivered, so this round must
    still be shown what is wrong and still pay for it — the advisory note changes
    nothing here, which is why this test passes before the fix and after it."""
    _one_tree(tmp_path)
    reason = PAST_EOF.format(line=500, eof=1, path="app/x.py")
    obj = _phantom_plus_met_obj(_honesty("advisory"))
    obj["clauses"][1] = {"clause": 2, "verdict": "unmet", "evidence_path": "app/x.py",
                         "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                         "how_verified": "ran", "note": "the second path is not covered"}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), contract=_TWO_CONTRACT)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False, detail
    assert data.get("external_blocker") is not True, (
        "the phantom citation swallowed the finding: the author would be told to "
        "re-gate and never shown what is wrong")
    assert data["review_retry"] is True and data["review_attempt"] == 1, data
    assert "the second path is not covered" in data["review_findings"], data
    assert reason in data["review_findings"], data
    ev = [e for e in events if e["event"] == "review"][-1]
    assert ev["blocking"] is True and ev["kind"] == "retry", ev
    phantom = next(c for c in ev["clauses"] if c["clause"] == 1)
    assert phantom["citation_only"] is True and phantom["downgraded"] == [reason], phantom


def test_a_note_denying_added_tests_against_a_positive_delta_spends_no_attempt(
        monkeypatch, tmp_path):
    """#1442 clause 4: "This round's diff adds no such test", said of a diff
    whose `def test_` delta over the base was four nodes. The deterministic half
    already counts them, so the contradiction is decidable without a model."""
    _one_tree(tmp_path)
    obj = {"premise": "sound", "summary": "", "clauses": [
        _phantom_entry(node="tests/test_x.py::test_it",
                       note="This round's diff adds no such test; the pin is a prior landing")],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["external_blocker"] is True
    assert "adds no test" in detail and "keeps its attempt" in detail
    assert events[-1]["blocking"] is False


def test_a_sound_premise_with_an_unmet_clause_is_a_retry_with_findings(monkeypatch, tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    obj = {"premise": "sound", "summary": "close",
           "clauses": [{"clause": 1, "verdict": "unmet", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                        "note": "only the chat path is covered"}],
           "test_honesty": [], "seams_unverified": ["loopback POST"]}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    g = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path)
    ok, detail, data = g.rung_review()
    assert ok is False and data["review_retry"] is True and data["review_attempt"] == 1
    assert "only the chat path is covered" in data["review_findings"]
    assert "seam unverified" in data["review_findings"]
    assert "fix what it names" in detail
    ev = events[-1]
    assert ev["event"] == "review" and ev["blocking"] and ev["kind"] == "retry"
    assert ev["clauses"][0]["verdict"] == "unmet"


def test_a_clause_the_grader_left_out_still_refuses_at_the_rung(monkeypatch, tmp_path):
    """One clause graded of two is a verdict on the change, not a rail failure.

    The unreadable-shape rail must not widen into an amnesty: a grader that
    answered clause 1 and said nothing about clause 2 still sends the round back
    with an attempt charged. Only the ALL-unusable case skips the judgment.
    """
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    contract = {"id": 7, "title": "t", "body": "b", "path": "",
                "clauses": ["clause one holds", "clause two holds"]}
    obj = {"premise": "sound", "summary": "one of two",
           "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                        "how_verified": "ran", "note": "ran it"}],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj), contract=contract)
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["review_retry"] is True and data["review_attempt"] == 1
    assert data.get("external_blocker") is not True, "an abstention is a judgment, not a rail"
    assert "not addressed by the grader" in data["review_findings"]
    assert "fix what it names" in detail
    ev = events[-1]
    assert ev["ok"] is True and ev["blocking"] and ev["kind"] == "retry"
    assert [c["verdict"] for c in ev["clauses"]] == ["met", "partial"]


def test_the_second_refusal_says_abort_and_the_third_never_asks_the_model(monkeypatch, tmp_path):
    obj = {"premise": "sound", "summary": "", "clauses": [
        {"clause": 1, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
         "test_node_id": "", "how_verified": "inferred", "note": "still"}],
        "test_honesty": [], "seams_unverified": []}
    grade = _grader(obj)
    _arm(monkeypatch, tmp_path, grade=grade, prior=1)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_attempt"] == 2 and "abort and report" in detail
    assert len(grade.calls) == 1
    _arm(monkeypatch, tmp_path, grade=grade, prior=2)
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_retry"] and data["review_exhausted"]
    assert len(grade.calls) == 1, "a third gate call must not spend another grading turn"


def test_an_unsound_premise_is_its_own_verdict(monkeypatch, tmp_path):
    obj = {"premise": "unsound", "summary": "the file it fixes was deleted in 5531f21",
           "clauses": [], "test_honesty": [], "seams_unverified": []}
    _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
    assert ok is False and data["review_premise_unsound"] is True
    assert "5531f21" in data["review_summary"] and "review_retry" not in data


def test_all_clauses_met_passes_and_names_the_session(monkeypatch, tmp_path):
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    obj = {"premise": "sound", "summary": "does what it says",
           "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                        "how_verified": "ran", "note": "ran it"}],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok and "1 met" in detail and data["review_session"] == "sess_r"
    assert events[-1]["blocking"] is False


def test_a_graded_refusal_records_the_clause_verdicts_it_refused_on(monkeypatch, tmp_path):
    """The PASS row wrote `clauses`; the refusal row wrote everything except them.

    #2448: `SM_20261008_232856` was aborted with the reason "clause 4 is not
    satisfied as written … that file is not in this diff", while both of its
    review attempts had graded all five clauses `met`. That sentence appears 0
    times in the round's own `gate.json`, and the review row in it — the artifact
    the round is told to read to explain its own refusal — carried the findings,
    the session, the attempt and the validated head, and no verdict for clause 4
    to contradict it with. `gate_ok: false` is consistent with that story: the
    gate did fail, just not the way the narrative said.
    """
    (tmp_path / "app").mkdir(); (tmp_path / "app" / "x.py").write_text("1\n")
    obj = {"premise": "sound", "summary": "close",
           "clauses": [{"clause": 1, "verdict": "unmet", "evidence_path": "app/x.py",
                        "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                        "note": "only the chat path is covered"}],
           "test_honesty": [], "seams_unverified": []}
    events = _arm(monkeypatch, tmp_path, grade=_grader(obj))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok is False and data["review_retry"] is True, detail
    assert [c["verdict"] for c in data["clauses"]] == ["unmet"], data.get("clauses")
    assert data["clauses"][0]["note"] == "only the chat path is covered"
    assert data["clauses"] == events[-1]["clauses"], \
        "the same rows the `review` ledger event already carries"

    # And the same SHAPE the PASS row writes, key for key: a refusal is not a
    # lesser record of the grade it just made.
    passed = {"premise": "sound", "summary": "does what it says",
              "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "app/x.py",
                           "evidence_line": 1, "test_node_id": "tests/test_x.py::test_it",
                           "how_verified": "ran", "note": "ran it"}],
              "test_honesty": [], "seams_unverified": []}
    _arm(monkeypatch, tmp_path, grade=_grader(passed))
    ok2, detail2, data2 = _Gate(7, ["app/x.py", "tests/test_x.py"], tmp_path).rung_review()
    assert ok2 is True and "1 met" in detail2, detail2
    assert sorted(data2["clauses"][0]) == sorted(data["clauses"][0])


def test_a_refusal_answered_from_the_ledger_replays_the_verdicts_with_its_findings(
        monkeypatch, tmp_path):
    """A cap-exhausted or same-head row quotes a grade the grader already made.

    It quotes the findings and, until #2448, dropped the verdicts that came with
    them — so a round told to abort by the per-round cap could read its own
    `gate.json` and find no clause verdict anywhere in it, exactly like the
    graded refusal. Measured over the 30 days to 2026-10-09: 157 refused review
    rows in `rounds/*/gate.json`, 9 cap-exhausted and 3 same-head among them, 0
    carrying `clauses`.
    """
    graded = [{"clause": 1, "verdict": "met", "how_verified": "ran"},
              {"clause": 2, "verdict": "unmet", "how_verified": "read"}]
    # The same commit twice: `head` cannot be resolved from a scratch worktree,
    # so the report carries it and the recorded refusal carries that value.
    _arm(monkeypatch, tmp_path, grade=_grader(None), prior=1,
         prior_head="b" * 40, prior_clauses=graded)
    same = _Gate(7, ["app/x.py"], tmp_path)
    same.report.head = "b" * 40
    ok, detail, data = same.rung_review()
    assert ok is False and data["review_same_head"] is True, detail
    assert data["clauses"] == graded, "the verdicts of the refusal it is quoting"

    # The per-round cap (2 graded refusals of distinct commits) and the hard cap
    # on grading turns both answer without a grading turn, and both replay the
    # LAST recorded refusal — findings and verdicts together.
    for prior in (2, 5):
        _arm(monkeypatch, tmp_path, grade=_grader(None), prior=prior, prior_clauses=graded)
        ok, detail, data = _Gate(7, ["app/x.py"], tmp_path).rung_review()
        assert ok is False and data["review_retry"] and data["review_exhausted"], detail
        assert data["clauses"] == graded, f"prior={prior}: {sorted(data)}"


def test_the_gate_event_carries_the_review_flags_and_findings(monkeypatch):
    """`gate.json` dies with the worktree; `implement_outcomes` reads these off
    the ledger long after."""
    events = []
    monkeypatch.setattr(G.S, "append_event", lambda e, **k: events.append(e))
    g = G.Gate("SM_FLAGS", ROOT, "HEAD")
    g._rung("review", lambda: (False, "sent back", {"review_retry": True,
                                                    "review_findings": "clause 1 unmet",
                                                    "review_attempt": 1}))
    g._rung("review", lambda: (False, "unsound", {"review_premise_unsound": True,
                                                  "review_summary": "false premise"}))
    assert events[0]["review_retry"] is True and events[0]["review_findings"] == "clause 1 unmet"
    assert events[0]["review_attempt"] == 1 and "review_premise_unsound" not in events[0]
    assert events[1]["review_premise_unsound"] is True and events[1]["review_summary"] == "false premise"


def test_preflight_refuses_a_contract_with_no_clauses_or_no_test(monkeypatch, tmp_path):
    """Zero-cost fail-fasts for a round that has a contract — cheaper to learn
    here than after a 77 s test run, and both are the round's own."""
    monkeypatch.setattr(RV, "item_contract", lambda iid, ledger=None: {
        "id": iid, "title": "t", "body": "", "clauses": [], "path": ""})
    g = G.Gate("SM_PF", ROOT, "HEAD", item_id=9)
    # Drive only the clause/test check by calling the same predicate preflight uses.
    contract = RV.item_contract(9)
    assert not contract["clauses"]
    src = (ROOT / "scripts" / "automod" / "gate.py").read_text()
    assert "has no acceptance clauses to judge" in src
    assert "nothing pins a clause" in src
    assert g.item_id == 9


def test_review_sits_after_tests_and_before_venv(monkeypatch):
    names: list[str] = []
    g = G.Gate("SM_POS", ROOT, "HEAD")
    monkeypatch.setattr(g, "_rung", lambda name, fn: names.append(name) or True)
    g.run()
    assert names.index("tests") < names.index("review") < names.index("venv")
    assert names.index("canary_smoke") == names.index("drill") - 1


# ── the backlog: retry, cap, disagreement, partial ─────────────────────────

def test_a_review_refusal_re_offers_the_item_with_the_findings_and_the_branch(isolated):
    write_item(isolated, 544, clauses=["a", "b"]); _confirm(544, clauses=["a", "b"])
    _round(544, "SM_1"); _review_refused("SM_1", 544, findings="clause 2 unmet: no loopback test")
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[544]
    assert verdict == "review_retry"
    assert "no loopback test" in detail and "automod/SM_1" in detail and "from_branch" in detail
    assert 544 not in B.implemented_ids(S.LEDGER_PATH)
    assert B.desired_statuses(S.LEDGER_PATH, None)[544][0] == "up_next"
    assert B.reoffer_reason(S.LEDGER_PATH, 544).startswith("review_retry:")


def test_the_review_re_offer_is_capped(isolated):
    """`REVIEW_RETRY_CAP` re-offers: the first round plus two more, each
    refused on a DIFFERENT clause (the same clause twice escalates earlier —
    see the next test). The fourth finished round is spent."""
    write_item(isolated, 545, clauses=["a", "b", "c", "d"]); _confirm(545, clauses=["a", "b", "c", "d"])
    for n, rid in enumerate(("SM_a", "SM_b", "SM_c"), 1):
        _round(545, rid); _review_refused(rid, 545, clauses_unmet=(n,), findings=f"f{n}")
        assert B.implement_outcomes(S.LEDGER_PATH)[545][0] == "review_retry", n
    _round(545, "SM_d"); _review_refused("SM_d", 545, clauses_unmet=(4,), findings="f4")
    verdict, _ = B.implement_outcomes(S.LEDGER_PATH)[545]
    assert verdict == "spent", "one round plus REVIEW_RETRY_CAP re-offers; then a human decides"
    assert B.desired_statuses(S.LEDGER_PATH, None)[545][0] == "draft"


def test_the_same_clause_twice_is_a_disagreement_and_escalates_early(isolated):
    """Author says met, grader says unmet, twice, on clause 1. A third round
    re-runs the argument; a human resolves it."""
    write_item(isolated, 546, clauses=["a", "b"]); _confirm(546, clauses=["a", "b"])
    _round(546, "SM_x"); _review_refused("SM_x", 546, clauses_unmet=(1,))
    assert B.review_disagreement(S.LEDGER_PATH, 546) is None
    _round(546, "SM_y"); _review_refused("SM_y", 546, clauses_unmet=(1, 2))
    assert B.review_disagreement(S.LEDGER_PATH, 546) == 1
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[546]
    assert verdict == "spent" and detail.startswith("review disagreement") and "clause 1" in detail
    # ...while different clauses on successive reviews are still a retry.
    write_item(isolated, 547, clauses=["a", "b"]); _confirm(547, clauses=["a", "b"])
    _round(547, "SM_p"); _review_refused("SM_p", 547, clauses_unmet=(1,))
    _round(547, "SM_q"); _review_refused("SM_q", 547, clauses_unmet=(2,))
    assert B.implement_outcomes(S.LEDGER_PATH)[547][0] == "review_retry"


def test_review_retry_yields_to_incomplete_and_supersedes_external(isolated):
    write_item(isolated, 548, clauses=["a"]); _confirm(548, clauses=["a"])
    _round(548, "SM_i", stop_reason="max_turns"); _review_refused("SM_i", 548)
    assert B.implement_outcomes(S.LEDGER_PATH)[548][0] == "incomplete"
    # A later passing gate on the same round supersedes the refusal.
    S.append_event({"event": "gate", "round_id": "SM_i", "rung": "drill", "ok": True}, path=S.LEDGER_PATH)
    assert "SM_i" not in B.review_retry_rounds(S.LEDGER_PATH)


def test_an_unsound_premise_spends_the_attempt(isolated):
    write_item(isolated, 549, clauses=["a"]); _confirm(549, clauses=["a"])
    _round(549, "SM_u")
    S.append_event({"event": "gate", "round_id": "SM_u", "rung": "review", "ok": False,
                    "detail": "review: premise unsound", "review_premise_unsound": True,
                    "review_summary": "false"}, path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[549][0] == "spent"
    assert "SM_u" in B.review_unsound_rounds(S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH, None, retriage_enabled=False)[549]
    assert want[0] == "draft" and want[2] is True
    want = B.desired_statuses(S.LEDGER_PATH, None)[549]
    assert want[0] == "draft" and len(want) == 2, "its one re-triage comes before a person"


def test_fresh_confirmations_are_picked_before_re_offers(isolated):
    """Oldest-first alone let a sent-back item monopolise the loop."""
    write_item(isolated, 550, clauses=["a"]); _confirm(550, clauses=["a"])     # older, sent back
    _round(550, "SM_o"); _review_refused("SM_o", 550)
    p = write_item(isolated, 551, clauses=["a"]); _confirm(551, clauses=["a"])  # newer, fresh
    fm, body = B._split_frontmatter(p.read_text())
    fm["created"] = datetime.now(timezone.utc).isoformat()
    p.write_text(f"---\n{yaml.dump(fm)}---\n{body}")
    B.reconcile_statuses(S.LEDGER_PATH, None)
    pair = B.select_confirmed(S.LEDGER_PATH, None)
    assert pair is not None and pair[0].id == 551
    # ...and a re-offer is still reachable once the fresh work is gone.
    B.set_status(551, "draft", "parked by a human")
    assert B.select_confirmed(S.LEDGER_PATH, None)[0].id == 550


# ── clause outcomes derive the acceptance; a nameless deferral is not_met ──

def test_parse_outcome_derives_the_acceptance_from_clauses():
    base = {"landed": True, "acceptance": "met", "deferred_to": [], "summary": "s", "spawned": []}
    all_met = [{"clause": 1, "outcome": "met", "evidence": "t::a", "deferred_to": []},
               {"clause": 2, "outcome": "met", "evidence": "t::b", "deferred_to": []}]
    assert B.parse_outcome({**base, "clause_outcomes": all_met})["acceptance"] == "met"
    one_unmet = all_met[:1] + [{"clause": 2, "outcome": "not_met", "evidence": "", "deferred_to": []}]
    out = B.parse_outcome({**base, "clause_outcomes": one_unmet})
    assert out["acceptance"] == "not_met" and B.unmet_clauses(out) == [2]
    deferred = all_met[:1] + [{"clause": 2, "outcome": "deferred", "evidence": "", "deferred_to": [618]}]
    out = B.parse_outcome({**base, "clause_outcomes": deferred})
    assert out["acceptance"] == "deferred" and out["deferred_to"] == [618]
    # `unnecessary` is a verdict on the item and is kept as stated.
    assert B.parse_outcome({**base, "acceptance": "unnecessary", "clause_outcomes": []})["acceptance"] == "unnecessary"


def test_a_deferral_that_names_nothing_is_not_met():
    """#544 exactly: `deferred`, `deferred_to: []`, and the item parked in
    `in_progress` with "close this when that closes" pointing at nothing."""
    base = {"landed": True, "acceptance": "deferred", "deferred_to": [], "summary": "s",
            "spawned": [], "clause_outcomes": []}
    assert B.parse_outcome(base)["acceptance"] == "not_met"
    nameless = [{"clause": 1, "outcome": "deferred", "evidence": "", "deferred_to": []}]
    assert B.parse_outcome({**base, "clause_outcomes": nameless})["acceptance"] == "not_met"
    assert B.parse_outcome({**base, "deferred_to": [7]})["acceptance"] == "deferred"
    assert "clause_outcomes" in B.IMPLEMENT_OUTCOME_SCHEMA["required"]
    assert B.IMPLEMENT_OUTCOME_SCHEMA["properties"]["clause_outcomes"]["items"]["properties"][
        "outcome"]["enum"] == list(B.CLAUSE_OUTCOMES)


def test_a_landed_round_with_an_unmet_clause_is_offered_once_more_for_it(isolated):
    p = write_item(isolated, 552, clauses=["a", "b"]); _confirm(552, clauses=["a", "b"])
    S.append_event({"event": "backlog_implement", "item_id": 552, "phase": "started"}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 552, "phase": "finished",
                    "round_id": "SM_l", "stop_reason": "stop", "num_turns": 30,
                    "outcome": {"landed": True, "acceptance": "not_met", "deferred_to": [],
                                "summary": "half", "spawned": [],
                                "clause_outcomes": [
                                    {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []},
                                    {"clause": 2, "outcome": "not_met", "evidence": "", "deferred_to": []}]}},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": "SM_l", "commit": "c0ffee00c0ffee"}, path=S.LEDGER_PATH)
    # Under observation: still in_progress.
    assert B.desired_statuses(S.LEDGER_PATH, None)[552][0] == "in_progress"
    S.append_event({"event": "settled", "commit": "c0ffee00c0ffee"}, path=S.LEDGER_PATH)
    out = B.close_settled_items(S.LEDGER_PATH, None)
    assert out == [{"item_id": 552, "closed": False, "acceptance": "not_met"}]
    assert "clause(s) [2]" in _fm(p)["activity_log"][-1]
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[552]
    assert verdict == "partial" and "[2]" in detail and "c0ffee00" in detail
    assert B.desired_statuses(S.LEDGER_PATH, None)[552][0] == "up_next"
    assert "partial" in I._reoffer_block(B.reoffer_reason(S.LEDGER_PATH, 552))


# ── clauses at triage: both parsers, the item, the fallback ─────────────────

TEXT = ("VERDICT: confirmed\nSURFACE: code\nCHECK: c\nEVIDENCE: e\n"
        "ACCEPTANCE: the retry fires once and the dashboard shows it\n"
        "ACCEPTANCE_CLAUSES:\n1. a retried item fires email_send once\n2. the counter is on /api/dashboard\n"
        "SPAWNED: none\n")


def test_the_text_path_splits_numbered_clauses_and_keeps_the_placeholder_rule():
    parsed = T.parse_verdict(TEXT)
    assert parsed["acceptance_clauses"] == ["a retried item fires email_send once",
                                           "the counter is on /api/dashboard"]
    assert parsed["acceptance"].startswith("the retry fires once")
    none = T.parse_verdict("VERDICT: stale\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: -\nACCEPTANCE_CLAUSES: none\n")
    assert none["acceptance_clauses"] == [] and none["acceptance"] == ""


def test_the_structured_path_carries_clauses_too():
    obj = {"verdict": "confirmed", "surface": "code", "check": "c", "evidence": "e",
           "acceptance": "x", "acceptance_clauses": ["one", "two", "none"], "spawned": []}
    assert T.parse_verdict("", obj)["acceptance_clauses"] == ["one", "two"]
    assert "acceptance_clauses" in B.TRIAGE_VERDICT_SCHEMA["required"]
    assert "ACCEPTANCE_CLAUSES:" in T.PROMPT


def test_record_verdict_writes_the_clauses_onto_the_item(isolated):
    p = write_item(isolated, 560, status="draft")
    item = next(i for i in B.open_items(None) if i.id == 560)
    B.record_verdict(item, "confirmed", "real", acceptance="x", acceptance_clauses=["one", "two"])
    fm = _fm(p)
    assert fm["acceptance_clauses"] == ["one", "two"] and fm["status"] == "up_next"
    assert "1. one" in p.read_text() and "graded one by one" in p.read_text()


def test_acceptance_clauses_fall_back_to_the_prose_for_old_items(isolated):
    write_item(isolated, 561); _confirm(561, acceptance="the check passes")
    contract = RV.item_contract(561)
    assert contract["clauses"] == ["the check passes"] and contract["title"] == "A thing"
    assert B.acceptance_clauses_of({"acceptance": "-"}) == []
    assert B.split_clause_lines("1. a\n2) b\n") == ["a", "b"]
    assert B.split_clause_lines("just prose") == ["just prose"]


# ── the implement side: prompt, resume, abort reason ───────────────────────

def test_the_implement_prompt_names_the_clauses_the_seams_and_the_item_id():
    text = I.PROMPT.format(item_id=9, status="up_next", priority="high", name="n", body="b",
                           triaged_ago="today", surface="code", check="c", evidence="e",
                           acceptance="a", clauses="    1. a\n    2. b", spawn_cap=I.SPAWN_CAP,
                           max_turns=I.DEFAULT_MAX_TURNS, gate_minutes=16, first_gate_by=23,
                           round_label="item9", reoffer="", members="", human_clauses="",
                           surface_rules="")
    assert "    1. a\n    2. b" in text
    # "Seams." and the review-rung procedure moved to the vault skill
    # `automod-change-own-code` (cut 4); the prompt names the skill instead.
    assert "item_id=9" in text and "automod-change-own-code" in text and "review rung" in text
    assert "A deferral that names no id is recorded as `not_met`" in text
    assert "per clause" in text


def test_the_reoffer_banner_for_a_review_retry_names_the_branch_to_resume():
    reason = ("review_retry: the review rung found the premise sound but the implementation or its "
              "tests short — clause 1 unmet; its work is on branch `automod/SM_20260910_070638` "
              "(pass it as from_branch)")
    banner = I._reoffer_block(reason)
    assert 'from_branch="automod/SM_20260910_070638"' in banner
    assert "never reached a verdict" not in banner
    assert "never reached a verdict" in I._reoffer_block("incomplete: ran out of clock")


@pytest.fixture()
def scratch(tmp_path, monkeypatch):
    live = tmp_path / "live"; (live / "app").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(live))
    git(live, "config", "user.email", "t@e.com"); git(live, "config", "user.name", "t")
    (live / "app" / "m.py").write_text("V = 1\n", encoding="utf-8")
    git(live, "add", "-A"); git(live, "commit", "-q", "-m", "base")
    for name in ("STATE_DIR", "BROKEN_DIR"):
        monkeypatch.setattr(S, name, tmp_path / "state")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "state" / "rounds")
    for name, fn in (("LEDGER_PATH", "promotions.jsonl"), ("HALTED_PATH", "halted"),
                     ("BROKEN_PATH", "BROKEN"), ("LOCK_PATH", "lock"), ("CURRENT_PATH", "current.json")):
        monkeypatch.setattr(S, name, tmp_path / "state" / fn)
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(R, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "LIVE_ROOT", live)
    monkeypatch.setattr(W, "WORK_ROOT", tmp_path / "work")
    return live


def test_a_round_can_resume_the_branch_review_sent_back(scratch, monkeypatch):
    """The re-offer is not a fresh start: the new worktree begins where the
    refused round left off, rebased onto live main, and the old branch is
    gone so branches do not accumulate forever."""
    # Round ids are second-resolution; two starts in one test need distinct ones.
    ids = iter(("SM_T1", "SM_T2"))
    monkeypatch.setattr(R, "_round_id", lambda: next(ids))
    first = R.start("first go", force=True, item_id=77)
    rid1 = first["round_id"]
    wt1 = Path(first["worktree"])
    (wt1 / "app" / "fix.py").write_text("FIX = 1\n", encoding="utf-8")
    git(wt1, "add", "-A"); git(wt1, "commit", "-q", "-m", "partial fix")
    spec1 = yaml.safe_load((S.ROUNDS_DIR / rid1 / "run_spec.yaml").read_text())
    assert spec1["item"] == {"id": 77}
    R.abort(rid1, reason="review sent it back: clause 1 unmet")
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_aborted"][-1]
    assert ev["reason"].startswith("review sent it back")
    assert W.branch_exists(scratch, f"automod/{rid1}")

    # main moves underneath in the meantime.
    (scratch / "app" / "other.py").write_text("O = 1\n", encoding="utf-8")
    git(scratch, "add", "-A"); git(scratch, "commit", "-q", "-m", "human commit")
    live_head = git(scratch, "rev-parse", "HEAD").stdout.strip()

    second = R.start("second go", force=True, item_id=77, from_branch=f"automod/{rid1}")
    try:
        wt2 = Path(second["worktree"])
        assert (wt2 / "app" / "fix.py").exists(), "the refused round's work is the starting point"
        assert (wt2 / "app" / "other.py").exists(), "rebased onto the moved main"
        assert second["base"] == live_head and second["rebased_onto"] == live_head
        assert second["from_branch"] == f"automod/{rid1}"
        assert not W.branch_exists(scratch, f"automod/{rid1}"), "the old branch is deleted"
        start_ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_start"][-1]
        assert start_ev["item_id"] == 77 and start_ev["from_branch"] == f"automod/{rid1}"
    finally:
        W.remove(second["round_id"], repo=scratch)


def test_an_abort_row_carries_the_gate_verdict_beside_the_reason(scratch, monkeypatch):
    """A reason that contradicts the artifact has to be visible in the row.

    #1169's round was aborted with the reason "Both review attempts spent …
    the review refused twice on clause 4/5 test honesty" 20 seconds after its
    `gate.json` reported `ok: true` with "review: 5 met of 5 clause(s)" — the
    narrative described the *first* gate run, and `abort()` then deleted the
    artifact that disproved it. Three implement-filed blockers (#1091, #1169,
    #1200) are the same shape. So the row carries the report's own verdict and
    head, read off disk before the round dir goes, and a reader needs no
    `gate.json` to spot the disagreement.
    """
    ids = iter(("SM_G1", "SM_G2"))
    monkeypatch.setattr(R, "_round_id", lambda: next(ids))

    gated = R.start("graded green", force=True, item_id=77)
    rid = gated["round_id"]
    (S.ROUNDS_DIR / rid).mkdir(parents=True, exist_ok=True)
    # A head no git command could answer for this round, so the row can only
    # have come from the file.
    (S.ROUNDS_DIR / rid / "gate.json").write_text(json.dumps(
        {"round_id": rid, "head": "deadbeefcafe1234", "ok": True,
         "rungs": [{"name": "review", "ok": True, "detail": "5 met of 5 clause(s)"}]}),
        encoding="utf-8")
    out = R.abort(rid, reason="the review refused twice on clause 4")
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "round_aborted"][-1]
    assert ev["gate_ok"] is True
    assert ev["gate_head"] == "deadbeefcafe1234"
    assert "refused twice" in ev["reason"]
    # and back to the caller that wrote the reason, at the moment it wrote it
    assert out["gate_ok"] is True and out["gate_head"] == "deadbeefcafe1234"

    # A round that never gated: the keys are present and empty, so the absence
    # of an artifact is a distinct answer from a gate that said no.
    dry = R.start("never gated", force=True, item_id=78)
    R.abort(dry["round_id"], reason="out of clock")
    ev2 = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "round_aborted"][-1]
    assert ev2["gate_ok"] == "" and ev2["gate_head"] == ""
    # #2448: the same present-and-empty answer for the clause verdicts, so a
    # reader never mistakes "no grade was recorded" for "nothing was met".
    assert ev2["review_clauses"] == ""


def _gate_json_for(rid, review_data, *, ok=False, head="deadbeefcafe1234"):
    """A `gate.json` in the shape the gate writes one: the overall verdict, and
    a `review` rung carrying the `data` that rung returned."""
    (S.ROUNDS_DIR / rid).mkdir(parents=True, exist_ok=True)
    (S.ROUNDS_DIR / rid / "gate.json").write_text(json.dumps(
        {"round_id": rid, "head": head, "ok": ok, "changed_paths": ["app/x.py"],
         "rungs": [{"name": "tests", "ok": True, "detail": "1234 passed"},
                   {"name": "review", "ok": False, "detail": "review sent it back",
                    "data": review_data}]}), encoding="utf-8")


def test_the_abort_row_carries_the_verdicts_that_refute_the_reason_beside_it(scratch, monkeypatch):
    """`gate_ok: false` does not contradict a clause-level story. #2448 clause 2.

    `SM_20261008_232856` was aborted at 00:09:45Z with "clause 4 is not satisfied
    as written: the clause names app/harness/tests/test_finalizer.py … but that
    file is not in this diff", 26 seconds after its second review attempt graded
    all five clauses `met` — clause 4 `how_verified: ran`, note "Ran
    app/harness/tests/test_finalizer.py: 45 passed". The blocker filed from that
    reason repeated the sentence. The stamp #1116 landed answered only "did the
    gate pass", which the story agrees with, so the row now carries the file's
    own clause summary too, read before the round dir is removed and never from
    the caller.
    """
    ids = iter(("SM_C1", "SM_C2"))
    monkeypatch.setattr(R, "_round_id", lambda: next(ids))
    rid = R.start("graded five of five", force=True, item_id=77)["round_id"]
    met = [{"clause": i, "verdict": "met", "how_verified": "ran"} for i in range(1, 6)]
    _gate_json_for(rid, {"review_retry": True, "review_attempt": 2,
                         "review_findings": "blocking test-honesty entry",
                         "clauses": met})
    out = R.abort(rid, reason="clause 4 is not satisfied as written")
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "round_aborted"][-1]
    assert ev["gate_ok"] is False and ev["gate_head"] == "deadbeefcafe1234"
    assert "clause 4 is not satisfied" in ev["reason"], "the narrative is kept, not edited"
    assert ev["review_clauses"] == "5 met of 5", "what the file the reason cites says"
    assert out["review_clauses"] == "5 met of 5", "and to the caller, at the moment it wrote it"

    # Partial verdicts keep their own counts rather than collapsing to a met
    # tally, so a row cannot read as "all met" when one clause did not pass.
    rid2 = R.start("graded four of five", force=True, item_id=78)["round_id"]
    _gate_json_for(rid2, {"review_retry": True, "clauses": [
        {"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "met"},
        {"clause": 3, "verdict": "partial"}, {"clause": 4, "verdict": "unmet"},
        {"clause": 5, "verdict": "post_landing"}]})
    R.abort(rid2, reason="out of clock")
    ev2 = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "round_aborted"][-1]
    assert ev2["review_clauses"] == "2 met, 1 partial, 1 unmet, 1 post_landing of 5"


def test_an_abort_row_leaves_absent_verdicts_absent_rather_than_zero_met(scratch, monkeypatch):
    """No grade, an unreadable report and an unevidenced refusal are one answer:
    no summary. #2448 clause 3.

    The distinction is what makes the stamp worth reading — `"0 met"` would be a
    verdict the grader never gave, and it would sit beside a reason that claims a
    clause went unmet as though it corroborated it.
    """
    ids = iter(("SM_N1", "SM_N2", "SM_N3", "SM_N4"))
    monkeypatch.setattr(R, "_round_id", lambda: next(ids))

    # (a) a round that never gated: no `gate.json` at all.
    never = R.start("never gated", force=True, item_id=77)["round_id"]
    R.abort(never, reason="out of clock")

    # (b) a report that cannot be parsed.
    broken = R.start("gate.json truncated", force=True, item_id=78)["round_id"]
    (S.ROUNDS_DIR / broken).mkdir(parents=True, exist_ok=True)
    (S.ROUNDS_DIR / broken / "gate.json").write_text('{"round_id": "SM_N2", "run', encoding="utf-8")
    R.abort(broken, reason="out of clock")

    # (c) a refusal row carrying findings but no clause verdicts — the shape
    # every one of the 157 refused rows in the 30 days to 2026-10-09 had.
    no_verdicts = R.start("refused without verdicts", force=True, item_id=79)["round_id"]
    _gate_json_for(no_verdicts, {"review_retry": True,
                                 "review_findings": "clause 2 unmet: no test"})
    R.abort(no_verdicts, reason="clause 2 unmet")

    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "round_aborted"]
    assert len(rows) == 3, [e.get("round_id") for e in rows]
    for ev in rows:
        assert "review_clauses" in ev, f"{ev['round_id']}: present like gate_ok"
        assert ev["review_clauses"] == "", f"{ev['round_id']}: and empty, not '0 met'"

    # And a grade that really did meet nothing is NOT the empty answer.
    none_met = R.start("graded nothing met", force=True, item_id=80)["round_id"]
    _gate_json_for(none_met, {"review_retry": True, "clauses": [
        {"clause": 1, "verdict": "unmet"}, {"clause": 2, "verdict": "unmet"}]})
    R.abort(none_met, reason="both clauses unmet")
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "round_aborted"][-1]
    assert ev["review_clauses"] == "2 unmet of 2"


def test_a_missing_resume_branch_falls_back_to_a_fresh_worktree(scratch):
    out = R.start("go", force=True, from_branch="automod/SM_NOPE")
    try:
        assert out["from_branch_missing"] is True and "does not exist" in out["note"]
        assert Path(out["worktree"]).exists()
    finally:
        W.remove(out["round_id"], repo=scratch)


# ── vault rounds: the same reader, before `git add` ────────────────────────

@pytest.fixture
def vault(tmp_path, monkeypatch):
    r = tmp_path / "obsidian"
    (r / "skills" / "foo").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    monkeypatch.setattr(V, "VAULT", r)
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    monkeypatch.setattr(V, "GRADER", None)
    return r


def _vault_events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


def test_a_vault_land_with_no_grader_records_why_it_was_not_reviewed(vault):
    """Was `assert _vault_events("vault_review") == []`: an abstention left no
    review event at all, and the landing said only `skipped`. #955's merged
    finding is that 34 of 54 item-bound landings looked like that — a grader
    outage, a surface mismatch and an unconsulted reviewer indistinguishable. An
    abstention still never blocks; it now says which of the six it was."""
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v2\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert out["review"] == "skipped" and _vault_events("vault_land")[-1]["review"] == "skipped"
    assert _vault_events("vault_land")[-1]["review_reason"] == out["review_reason"]
    ev = _vault_events("vault_review")[-1]
    assert ev["blocking"] is False and ev["kind"] == "skipped"
    assert "no grader configured" in ev["review_reason"]


def test_a_vault_review_refusal_leaves_the_edit_then_reverts_on_the_second(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("retry", "clause 1 unmet: the skill never says when"))
    S.append_event({"event": "backlog_implement", "item_id": 9, "phase": "started"}, path=S.LEDGER_PATH)
    f = vault / "skills" / "foo" / "SKILL.md"
    f.write_text("---\nname: foo\n---\n# foo v2\n")
    with pytest.raises(V.VaultRoundError, match="still in place"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert "v2" in f.read_text(), "first refusal: the model fixes it in place"
    ev = _vault_events("vault_review")[-1]
    assert ev["blocking"] and ev["attempt"] == 1 and ev["review_retry"] and ev["reverted"] == []
    with pytest.raises(V.VaultRoundError, match="reverted"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert f.read_text() == "---\nname: foo\n---\n# foo\n", "second refusal: put back"
    assert _vault_events("vault_review")[-1]["reverted"] == ["skills/foo/SKILL.md"]


def test_an_unsound_vault_premise_reverts_at_once(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("unsound", "the task this skill serves was deleted"))
    f = vault / "skills" / "foo" / "SKILL.md"
    f.write_text("---\nname: foo\n---\n# foo v3\n")
    with pytest.raises(V.VaultRoundError, match="premise unsound"):
        V.land(["skills/foo/SKILL.md"], "skill: foo v3", item_id=9)
    assert "v3" not in f.read_text()
    assert _vault_events("vault_review")[-1]["review_premise_unsound"] is True


def test_a_vault_grader_that_blows_up_never_blocks_a_validated_landing(vault, monkeypatch):
    def boom(**kw):
        raise RuntimeError("engine gone")
    monkeypatch.setattr(V, "GRADER", boom)
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v4\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v4", item_id=9)
    assert out["ok"] and out["review"] == "skipped"


def test_a_vault_land_passing_review_records_it(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("pass", "all clauses met"))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v5\n")
    out = V.land(["skills/foo/SKILL.md"], "skill: foo v5", item_id=9)
    assert out["review"] == "pass"
    assert _vault_events("vault_review")[-1]["blocking"] is False


def test_the_vault_grader_only_judges_a_vault_items_clauses(isolated, monkeypatch):
    """#551 was a `code` item whose round landed a skill and a task file first.
    The vault grader held that half to the whole contract and refused it for
    the code it had not written yet — and would have every time. Only a
    `vault` surface's clauses can be met by vault paths; the rest are the code
    gate's. No network: the skip is decided before any grader runs."""
    write_item(isolated, 570, clauses=["a"])
    S.append_event({"event": "backlog_triage", "item_id": 570, "verdict": "confirmed",
                    "surface": "code", "acceptance": "a", "acceptance_clauses": ["a"]},
                   path=S.LEDGER_PATH)
    monkeypatch.setattr(RV, "run_grader", lambda **kw: pytest.fail("grader must not run"))
    kind, why, clauses = RV.grade_vault(item_id=570, paths=["skills/x/SKILL.md"], diff="+x",
                                        vault=isolated)
    assert kind == "skipped" and "surface is code" in why and clauses == []
    S.append_event({"event": "backlog_triage", "item_id": 570, "verdict": "confirmed",
                    "surface": "vault", "acceptance": "a", "acceptance_clauses": ["a"]},
                   path=S.LEDGER_PATH)
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {
        "premise": "sound", "summary": "ok", "test_honesty": [], "seams_unverified": [],
        "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
                     "evidence_line": 1, "test_node_id": "", "how_verified": "read", "note": ""}]}})
    (isolated / "skills" / "x").mkdir(parents=True); (isolated / "skills" / "x" / "SKILL.md").write_text("x")
    assert RV.grade_vault(item_id=570, paths=["skills/x/SKILL.md"], diff="+x", vault=isolated)[0] == "pass"


# ── #955: the landing clause cannot be graded before the commit ─────────────

# #972 clause 11 and #993 clause 10, verbatim from their front matter. The two
# rounds with a recorded death are #425 (refused twice, clause 6 both times, work
# reverted at attempt 2) and #502 (attempt 1, clause 6, "no revertable sha").
# Triage listed nine open items carrying such a clause; four name the artefact.
LANDING_CLAUSE_972 = ("The change lands via automod_vault_land as one revertable sha "
                      "on vault main, commit message naming #972")
LANDING_CLAUSE_993 = ("The change touches only vault paths — no file under ~/lloyd "
                      "modified — and lands through automod_vault_land as a revertable sha.")
# And a clause that only mentions the landing in passing, plus one whose author
# already said which half the reviewer is to grade (#463 clause 4's shape).
CONTENT_CLAUSES = ["the skill names the retry rule",
                   "No field list names `board_id` (0 occurrences across 37 real keys).",
                   "The commit message names the item, and the change was reviewed before commit."]


def _vault_contract(clauses):
    return {"id": 1, "title": "t", "body": "b", "clauses": list(clauses), "members": [],
            "path": "", "amendments": [], "human_clauses": []}


def test_a_landing_clause_is_the_only_kind_of_clause_the_vault_grader_recognises():
    """Narrow on purpose: a false positive takes a gradeable clause out of the
    reviewer's hands, which is worse than the artifact it fixes."""
    graded = RV.landing_clause_indices(CONTENT_CLAUSES + [LANDING_CLAUSE_972, LANDING_CLAUSE_993])
    assert graded == [4, 5]
    # An author-drawn line stays with the reviewer (#463 clause 4, amended by hand).
    claimed = ("Pre-landing, graded by this review: both files are modified in the "
               "working tree on branch main. Post-landing, the sha.")
    assert RV.landing_clause_indices([claimed]) == []
    # The other spellings the triage listed on #955.
    assert RV.landing_clause_indices(["The change is submitted through "
                                      "`automod_vault_land` as one call naming exactly the "
                                      "six SKILL.md paths"]) == [1]
    assert RV.landing_clause_indices(["The change lands through automod_vault_land "
                                      "naming exactly the touched paths"]) == [1]
    assert RV.landing_clause_indices([]) == [] and RV.landing_clause_indices(None) == []


def test_the_vault_prompt_tells_the_reviewer_the_commit_is_pending(isolated):
    """#425 attempt 1: "Nothing landed … No new sha" — the reviewer ran `git log`
    in the vault and refused a round for the state it was standing in. The prompt
    has to say the commit is the CONSEQUENCE of a pass."""
    p = RV.build_vault_prompt(contract=_vault_contract(["content", LANDING_CLAUSE_972]),
                              paths=["skills/x/SKILL.md"], diff="+x", vault=isolated,
                              landing_clauses=[2])
    assert "The commit does not exist while you grade." in p
    assert ("on a pass the caller commits exactly these paths on the vault's `main` "
            "as one revertable sha") in p
    assert "Clauses 2 have the landing itself as their subject" in p
    assert "Never refuse a round because no sha exists yet" in p
    assert "never report `git log` as evidence that the change did not land" in p
    assert "The caller records their verdict from the commit it creates" in p
    # The ordering is stated for every vault round; the per-clause note only where
    # a clause actually needs it.
    plain = RV.build_vault_prompt(contract=_vault_contract(["content only"]),
                                 paths=["a.md"], diff="+x", vault=isolated)
    assert "The commit does not exist while you grade." in plain
    assert "have the landing itself as their subject" not in plain


def test_a_landing_clause_cannot_refuse_a_round_a_content_clause_still_can(isolated, monkeypatch):
    """The exclusion must not weaken the guard for anything else: clause 1 is a
    content clause the grader says is unmet, so the round still comes back —
    with the landing clause advisory, and marked so the caller can grade it.

    What is policy-free is the landing clause's side: `grade_vault` rewrites
    that verdict to `post_landing` before `decide` sees it. (This ran under the
    `table` policy until it was retired on 2026-09-24; an unmet clause refuses
    under the grader policy all the same.)"""
    write_item(isolated, 573, clauses=["the skill names the retry rule", LANDING_CLAUSE_972])
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {
        "premise": "sound", "summary": "s", "test_honesty": [], "seams_unverified": [],
        "clauses": [{"clause": 1, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
                     "test_node_id": "", "how_verified": "read", "note": "no retry rule"},
                    {"clause": 2, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
                     "test_node_id": "", "how_verified": "read", "note": "no sha yet"}]}})
    kind, why, rows = RV.grade_vault(item_id=573, paths=["skills/x/SKILL.md"], diff="+x",
                                     vault=isolated)
    assert kind == "retry"
    assert "clause 1 unmet" in why, "an ordinary content clause still refuses"
    assert "clause 2 unmet" not in why, "the landing clause refused it — the #955 artifact"
    assert "observable only after landing" in why
    assert rows[1] == {"clause": 2, "verdict": "post_landing", "subject": "landing"}
    assert rows[0]["verdict"] == "unmet"


def test_a_passing_vault_review_returns_the_landing_clause_for_the_caller_to_close(isolated, monkeypatch):
    """All the content clauses met and the landing clause not gradable: that is a
    pass, and the row the caller rewrites is marked rather than silently `met`."""
    write_item(isolated, 574, clauses=["the skill names the retry rule", LANDING_CLAUSE_993])
    (isolated / "skills" / "y").mkdir(parents=True)
    (isolated / "skills" / "y" / "SKILL.md").write_text("---\nname: y\n---\n# y\n")
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {
        "premise": "sound", "summary": "s", "test_honesty": [], "seams_unverified": [],
        "clauses": [{"clause": 1, "verdict": "met", "evidence_path": "skills/y/SKILL.md",
                     "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                     "note": "named"},
                    {"clause": 2, "verdict": "unmet", "evidence_path": "", "evidence_line": 0,
                     "test_node_id": "", "how_verified": "read", "note": "no commit"}]}})
    kind, why, rows = RV.grade_vault(item_id=574, paths=["skills/y/SKILL.md"], diff="+x",
                                     vault=isolated)
    assert kind == "pass", why
    assert rows[1] == {"clause": 2, "verdict": "post_landing", "subject": "landing"}
    assert rows[0]["verdict"] == "met"


def test_the_vault_grader_names_the_way_it_abstains(isolated, monkeypatch):
    """Six causes, six wordings. #955's merged finding: 34 of 54 item-bound
    successful landings recorded one unlabelled `skipped`, so a refusal a later
    attempt ignored was indistinguishable from a grader outage."""
    write_item(isolated, 575, clauses=["a"])
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": False, "error": "backend 503 after 420s"})
    assert RV.grade_vault(item_id=575, paths=["a.md"], diff="+x", vault=isolated)[1] == (
        "grader did not answer: backend 503 after 420s")
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": True, "structured": {"premise": "sideways"}})
    assert RV.grade_vault(item_id=575, paths=["a.md"], diff="+x", vault=isolated)[1] == (
        "grader returned an unusable object")
    write_item(isolated, 576)
    assert RV.grade_vault(item_id=576, paths=["a.md"], diff="+x", vault=isolated)[1] == (
        "item #576 has no acceptance clauses")


def test_evidence_paths_are_normalized_before_they_are_judged(wt, tmp_path):
    """The schema asks for a bare worktree-relative file; the grader writes
    `app/x.py:164`, `a.py:224,253,201-214`, `~/vault/SOUL.md + ~/vault/b.md (…)`.
    Every one of the first four backfill rows had a real `met` downgraded for
    it. A path that exists, in any of those spellings, stands."""
    n = RV.normalize_evidence_path
    assert n("app/x.py", wt) == "app/x.py"
    assert n("app/x.py:12", wt) == "app/x.py"
    assert n("app/x.py:12,40-52", wt) == "app/x.py"
    assert n("./app/x.py:3", wt) == "app/x.py"
    assert n("`app/x.py:3`", wt) == "app/x.py"
    outside = tmp_path / "vault" / "SOUL.md"; outside.parent.mkdir(); outside.write_text("s")
    assert n(f"{outside} + {outside.parent}/other.md (trim is vault-side)", wt) == str(outside)
    assert n("app/nope.py:1", wt) == "" and n("", wt) == "" and n("/nope/x.md", wt) == ""
    parsed = RV.parse_review(_obj(evidence_path="app/x.py:3,7"), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "met" and parsed["clauses"][0]["evidence_path"] == "app/x.py"


# ── #1254: the cited line is bounded like the cited path ────────────────────

def test_a_met_citing_a_line_past_eof_is_downgraded(wt):
    """`SM_20260914_145943` cl.5 cited `fixture_iv_loop_turn.py:1007` in a
    52-line file and passed: the path rail resolved the file and nothing read
    the number. The clause is downgraded and the citation recorded, the way an
    unresolvable path is."""
    parsed = RV.parse_review(_obj(evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and parsed["downgraded"] == [1]
    assert c["evidence_path"] == "app/x.py" and c["evidence_line"] == 500
    assert any("past EOF" in w and "(6 lines)" in w for w in c["downgraded"]), c
    assert any("evidence_line 500 is past EOF of app/x.py" in u for u in c["citation_unresolved"])
    assert "accepted" not in c
    # #1750 overturns the ruling this file recorded here for a year ("the file is
    # real, the number is wrong — the review stays actionable and the clause is
    # simply unmet"). Round SM_20260928_164226 is the counter-evidence: the grader
    # cited line 769 of a 366-line file, its own finding text affirmed the clause, and
    # the round spent one of its two review attempts on a number only the grader could
    # have written. An impossible line in a file that DID resolve is the grader's
    # arithmetic. Where the clause has a defect besides that number it stays a graded
    # refusal — see
    # `test_a_line_past_eof_beside_a_failing_node_rail_is_still_a_graded_refusal`.
    assert c["downgraded"] == ["evidence_line 500 past EOF (6 lines) of app/x.py"]
    assert any("past EOF" in u for u in parsed["unreliable"]), parsed["unreliable"]


def test_a_line_inside_the_file_and_the_last_line_stand(wt):
    for line in (3, 6):
        parsed = RV.parse_review(_obj(evidence_line=line), worktree=wt,
                                 changed_tests=["tests/test_x.py"], n_clauses=1)
        assert parsed["clauses"][0]["verdict"] == "met" and parsed["downgraded"] == [], line
    # Line 0 is the schema's "no line" and is never a claim.
    parsed = RV.parse_review(_obj(evidence_line=0), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "met"


def test_a_past_eof_line_is_recorded_on_a_non_met_verdict_too(wt):
    parsed = RV.parse_review(_obj(verdict="partial", evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and "downgraded" not in c
    assert any("past EOF" in u for u in c["citation_unresolved"])


def test_the_line_is_bounded_only_against_the_file_the_grader_named_first(wt):
    """#1252 widened the path rail to later tokens; `evidence_line` still
    describes the first one, so a citation resolved through its second token
    is not held to a number written for another file."""
    parsed = RV.parse_review(_obj(evidence_path="scripts/nope.py:3; tests/test_x.py:1",
                                  evidence_line=3),
                             worktree=wt, changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "met" and c["evidence_path"] == "tests/test_x.py"
    assert "citation_unresolved" not in c


def test_a_vault_path_outside_the_worktree_is_not_line_checked(wt, tmp_path_factory):
    """The worktree is the commit under review; a vault file is live and
    shared, and a line into it is not a claim about the commit."""
    outside = tmp_path_factory.mktemp("vault") / "SOUL.md"
    outside.write_text("one line\n")
    parsed = RV.parse_review(_obj(evidence_path=f"{outside}:500", evidence_line=500), worktree=wt,
                             changed_tests=["tests/test_x.py"], n_clauses=1)
    c = parsed["clauses"][0]
    assert c["verdict"] == "met" and c["evidence_path"] == str(outside)
    assert "citation_unresolved" not in c
    assert RV.evidence_line_past_eof(str(outside), 500, wt) is None
    assert RV.evidence_line_past_eof("app/x.py", 7, wt) == 6
    assert RV.evidence_line_past_eof("app/x.py", 6, wt) is None
    assert RV.evidence_line_past_eof("app/nope.py", 1, wt) is None


# ── #2083: a line the clause's OWN node file contains is not a fabricated citation ──
#
# A clause names TWO files — `evidence_path`, and the file its `test_node_id`
# points into — and `evidence_line` describes only the first. A grader citing
# `tests/test_intel_pipeline_scorer.py:2474` while `evidence_path` holds the
# fixture that node reads has mispaired its citation, and until #2083 the line
# rail bounded the number against the SHORTER file: the clause went `met`→
# `partial`, was marked `citation_only`, the whole review came back `unreliable`,
# and a full grading turn went with it (median 271 s across the 33 past-EOF
# review events in `promotions.jsonl`). The widening honours the number against
# the node file; the tests below hold the three shapes it must NOT swallow — a
# line neither file can contain, a node file the diff never touched, and a node
# that names no countable test file at all.

@pytest.fixture
def wt_pair(tmp_path):
    """A worktree whose two cited files differ in length, which is the whole
    point of it: `app/x.py` is SIX lines and `tests/test_pair.py` is TWELVE, so
    line 10 is inside the node file and past EOF of the paired path. The line
    counts below are this fixture's, and the assertions quote them."""
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "x.py").write_text("1\n" * 6)
    (tmp_path / "tests" / "test_pair.py").write_text(
        "\n".join(f"# l{i}" for i in range(1, 13)) + "\n")
    return tmp_path


def _mispaired_obj(**clause):
    """A `met` clause in the `wt_pair` shape: line 10, `evidence_path` the 6-line
    `app/x.py`, `test_node_id` inside the 12-line `tests/test_pair.py`. Every
    other rail holds, so the line is the only thing in dispute."""
    base = {"clause": 1, "verdict": "met", "evidence_path": "app/x.py", "evidence_line": 10,
            "test_node_id": "tests/test_pair.py::test_it", "how_verified": "ran", "note": "ok"}
    base.update(clause)
    return {"premise": "sound", "clauses": [base], "test_honesty": [], "seams_unverified": [],
            "summary": "fine"}


# The diff of a round that changed the implementation AND its test — the shape
# the honour is scoped to.
PAIR_DIFF = {"changed_tests": ["tests/test_pair.py"],
             "changed_paths": ["app/x.py", "tests/test_pair.py"]}


def test_a_line_inside_the_node_file_is_graded_met_though_past_the_paired_path(wt_pair):
    """Clause 1: the line is past EOF of the paired `evidence_path` but inside the
    12-line file the clause's own node names, and that file is in the diff, so the
    clause stands `met` and the review grades the diff.

    The demotion this removes is the expensive one: `citation_only` routes the
    clause to the #1750 re-ask, `unreliable` means nothing on the diff was graded,
    and the round pays a grading turn to find out the number was fine all along.
    Line 12 is the node file's last line and is honoured too — the bound is
    inclusive, and an off-by-one here is the same wasted turn.
    """
    for line in (10, 12):
        parsed = RV.parse_review(_mispaired_obj(evidence_line=line), worktree=wt_pair,
                                 n_clauses=1, tests_passed=True, **PAIR_DIFF)
        c = parsed["clauses"][0]
        assert c["verdict"] == "met", (line, c)
        assert "citation_only" not in c, (line, c)
        assert "downgraded" not in c, (line, c)
        assert "citation_unresolved" not in c, (line, c)
        assert parsed["downgraded"] == [], (line, parsed["downgraded"])
        assert parsed["unreliable"] == [], (line, parsed["unreliable"])


def test_a_line_past_eof_of_both_named_files_is_still_fabricated_support(wt_pair):
    """Clause 2: the #1254 rail is intact. 999 is past EOF of `app/x.py` AND past
    EOF of the 12-line node file, so nothing about the widening may excuse it — the
    clause keeps its `partial`, its `citation_only` mark, and the verbatim reason
    that 15 rounds of fabricated line numbers were caught by.

    Line 13 is the other half of the inclusive bound from clause 1: one past the
    node file's end, and still refused. The `{eof}` in the reason stays the count
    of the paired path (6), because that is the file the number was cited against.
    """
    for line in (13, 999):
        parsed = RV.parse_review(_mispaired_obj(evidence_line=line), worktree=wt_pair,
                                 n_clauses=1, tests_passed=True, **PAIR_DIFF)
        c = parsed["clauses"][0]
        assert c["verdict"] == "partial", (line, c)
        assert c.get("citation_only") is True, (line, c)
        assert c["downgraded"] == [PAST_EOF.format(line=line, eof=6, path="app/x.py")], \
            (line, c)
        assert parsed["downgraded"] == [1], (line, parsed["downgraded"])


def test_the_honour_does_not_fire_for_a_node_file_the_diff_never_touched(wt_pair):
    """Clause 3: the widening is scoped to files this round changed. Here the diff
    is only `app/x.py`, so line 10 sits inside a test file the round never touched
    — and a number into an untouched file is a claim about a file nobody is
    grading on this round's authority, so the clause is still demoted.

    `unreliable` stays empty, which is the half that makes this a grade: the node
    rail has its own reason to speak here (`how_verified: read` never carries an
    existing test outside the diff — the suite was not measured), so the clause is
    demoted on findings the author can act on rather than routed to #1750's
    re-ask. If the honour had been scoped only by file LENGTHS and not by the
    diff, the past-EOF sentence would be gone from `downgraded` even though the
    verdict stayed `partial` — so the reason text is what this assertion pins.
    `accepted` is absent outright, which is the strong form: `parse_review` only
    writes the key when a waiver exists, so an empty list and a missing key are the
    same record and a populated one would mean the honour had fired.
    """
    parsed = RV.parse_review(_mispaired_obj(how_verified="read"), worktree=wt_pair,
                             n_clauses=1, tests_passed=True,
                             changed_tests=[], changed_paths=["app/x.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", c
    assert PAST_EOF.format(line=10, eof=6, path="app/x.py") in c["downgraded"], c
    assert c.get("citation_only") is not True, c
    assert parsed["unreliable"] == [], parsed["unreliable"]
    assert "accepted" not in c, c


def test_a_lone_past_eof_line_beside_an_untouched_node_file_still_routes_to_re_ask(
        wt_pair):
    """The other shape of the same scope guard: when the node IS the
    existing-test-outside-the-diff shape the node rail accepts (a suite run that
    passed), the past-EOF line is again un-honoured — and now it is the clause's
    ONLY defect, so #1750's flag and its re-ask sentence apply exactly as they did
    before this round. The scope guard removes honour, never routing: a number
    into a file the diff never touched is still a grader citation, and an
    untouched node file is still not the author's to fix by editing code.
    """
    parsed = RV.parse_review(_mispaired_obj(), worktree=wt_pair, n_clauses=1,
                             tests_passed=True, changed_tests=[], changed_paths=["app/x.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial" and c.get("citation_only") is True, c
    assert c["downgraded"] == [PAST_EOF.format(line=10, eof=6, path="app/x.py")], c
    assert "accepted" not in c, c
    assert len(parsed["unreliable"]) == 1 and "clause 1" in parsed["unreliable"][0], \
        parsed["unreliable"]


@pytest.mark.parametrize("node_kind,changed_tests,how_verified,tests_passed", [
    # A suite-level run. `tests/ -k test_pair` names a directory, so
    # `_test_file_cited` rightly returns "" (reading its first word as a path
    # would invent a phantom for an honest answer) — and the node rail still
    # holds because the tests rung passed, which makes the line the clause's
    # ONLY defect and the flag the correct mark.
    ("suite", ["tests/test_pair.py"], "ran", True),
    # A node whose file part IS shaped like a test file, in the diff, but is not
    # on disk at the graded head. This is the clause's whole point: the honour is
    # a POSITIVE read of a line count, never "the rail did not say past EOF".
    # `evidence_line_past_eof` answers None for unreadable, outside-the-worktree
    # and in-range alike, so an absent file would sail through on that reading.
    ("absent", ["tests/test_gone.py"], "ran", True),
])
def test_a_node_that_names_no_countable_test_file_never_rescues_a_past_eof_line(
        wt_pair, node_kind, changed_tests, how_verified, tests_passed):
    """Clause 4: with no countable file inside the diff, line 10 stays past EOF of
    `app/x.py`, and in both shapes the line is the clause's only defect, so it
    still carries `citation_only` exactly as it did before the widening.
    """
    node = {"suite": "tests/ -k test_pair",
            "absent": "tests/test_gone.py::test_it"}[node_kind]
    parsed = RV.parse_review(_mispaired_obj(test_node_id=node, how_verified=how_verified),
                             worktree=wt_pair, n_clauses=1, tests_passed=tests_passed,
                             changed_tests=changed_tests,
                             changed_paths=["app/x.py"] + list(changed_tests))
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", c
    assert PAST_EOF.format(line=10, eof=6, path="app/x.py") in c["downgraded"], c
    assert c.get("citation_only") is True, (node_kind, c)
    assert "accepted" not in c, (node_kind, c)


def test_a_node_naming_a_test_file_outside_the_worktree_is_refused_by_both_rails(
        wt_pair, tmp_path_factory):
    """Clause 4's other shape, recorded with the outcome it actually has.

    A node naming a test file that exists one directory OUT of the worktree is the
    case where the honour's two conditions come apart: `_test_file_cited` hands
    back that absolute path (it is on disk, so it is no phantom) and line 10 really
    is inside it, so only the diff-scope condition can decline it. The node rail
    declines independently — a file outside the commit under review is not
    #487's existing-test shape either — so the clause carries TWO reasons and
    #1750's single-reason guard leaves `citation_only` unset. That is the older
    ruling holding: a clause with a defect the author can act on is a graded
    refusal, not a free re-ask. `accepted` is cleared with `why`, so no waiver of
    the line rail is recorded either way.
    """
    outside = tmp_path_factory.mktemp("outside-2083") / "test_x.py"
    outside.write_text("\n".join(f"# l{i}" for i in range(1, 13)) + "\n")
    parsed = RV.parse_review(_mispaired_obj(test_node_id=f"{outside}::test_it"),
                             worktree=wt_pair, n_clauses=1, tests_passed=True,
                             changed_tests=["tests/test_pair.py"],
                             changed_paths=["app/x.py", "tests/test_pair.py"])
    c = parsed["clauses"][0]
    assert c["verdict"] == "partial", c
    assert PAST_EOF.format(line=10, eof=6, path="app/x.py") in c["downgraded"], c
    assert any("not in a test file this diff changed" in d for d in c["downgraded"]), c
    assert c.get("citation_only") is not True, c
    assert "accepted" not in c, c
    # The number really is inside that file — 12 lines, line 10 — so what refused
    # it is the two rails, not the arithmetic. `worktree_line_count` is not the
    # witness: a file outside the worktree is exactly what it refuses to count.
    assert len(outside.read_text().splitlines()) == 12, "line 10 is inside it"
    assert RV.worktree_line_count(str(outside), wt_pair) is None


def test_a_honoured_line_is_recorded_under_accepted_naming_the_file_it_was_read(wt_pair):
    """Clause 5: a waived rail is a decision, so the clause says which file the
    number was bounded against and how long that file was at the graded head, and
    the top-level `downgraded` does not name the clause at all.

    Without this the honour would be a silent exemption: `gate.json` would show a
    clean `met` and a reader could not tell a genuinely short path from a line
    rescued by the widening — which is exactly how the #1254 fabrications hid.
    """
    parsed = RV.parse_review(_mispaired_obj(), worktree=wt_pair, n_clauses=1,
                             tests_passed=True, **PAIR_DIFF)
    c = parsed["clauses"][0]
    assert c["verdict"] == "met", c
    assert len(c.get("accepted") or []) == 1, c
    honour = c["accepted"][0]
    assert honour.startswith("evidence_line 10 read against tests/test_pair.py "), honour
    assert "12 lines at the graded head" in honour, honour
    assert "test_node_id" in honour, honour
    assert parsed["downgraded"] == [], parsed["downgraded"]
    assert 1 not in parsed["downgraded"], parsed["downgraded"]


def test_worktree_line_count_counts_only_files_inside_the_worktree(wt_pair, tmp_path_factory):
    """The helper behind clause 4, on its own terms: a number, or None when there
    is no number to have — never a None that a caller can misread as `in range`.

    The outside case is a file that EXISTS, is readable and has lines to count, one
    directory out. Asserting None for a path that names nothing at all would pass
    whatever the boundary code said, which is the difference between pinning the
    worktree rail and pinning `is_file()`.
    """
    outside = tmp_path_factory.mktemp("outside-count") / "test_x.py"
    outside.write_text("1\n2\n3\n")
    assert RV.worktree_line_count("tests/test_pair.py", wt_pair) == 12
    assert RV.worktree_line_count("app/x.py", wt_pair) == 6
    assert RV.worktree_line_count(str(outside), wt_pair) is None
    assert RV.worktree_line_count("app/nope.py", wt_pair) is None
    assert RV.worktree_line_count("", wt_pair) is None
    assert RV.worktree_line_count("../outside.py", wt_pair) is None


def test_a_mispaired_citation_no_longer_sends_the_round_back_for_a_re_gate(
        monkeypatch, tmp_path):
    """The seam the whole item is about: `Gate.rung_review` is what consumes
    `parse_review` in production (`scripts/automod/gate.py:2265`), and before
    #2083 the mispaired clause came back through it as `unreliable` — an
    `external_blocker` that tells the author to re-gate and re-runs the grader
    for a number that was never wrong.

    Same tree as `wt_pair`, built here because the rung takes `tmp_path` as its
    worktree. The honoured clause now walks the rung's clean-pass arm: the rung
    passes, the round is not asked to re-gate, and the ledger row carries the
    honour on the clause so the waiver is in the record and not only in the
    verdict. Without the widening this is `ok is False` with
    `external_blocker: True`, which is what makes this a test of the seam and
    not of the fixture.
    """
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "x.py").write_text("1\n" * 6)
    (tmp_path / "tests" / "test_pair.py").write_text(
        "\n".join(f"# l{i}" for i in range(1, 13)) + "\n")
    events = _arm(monkeypatch, tmp_path, grade=_grader(_mispaired_obj()))
    ok, detail, data = _Gate(7, ["app/x.py", "tests/test_pair.py"],
                             tmp_path).rung_review()
    assert ok is True, (detail, data)
    assert "1 met of 1" in detail, detail
    assert data.get("external_blocker") is not True, data
    assert "review_retry" not in data, "an honoured citation is not a re-ask"
    ev = events[-1]
    assert ev["event"] == "review" and ev["ok"] is True and ev["kind"] == "pass", ev
    assert ev["downgraded"] == [], ev
    assert any("read against tests/test_pair.py" in a
               for a in ev["clauses"][0].get("accepted") or []), ev


# ── a `landed: true` the ledger cannot see is not stored as landed ──────────
#
# `landed` came straight out of the implementer's structured self-report
# (`parse_outcome`, then the `finished` row) with nothing comparing it to the
# ledger. Measured on 2026-09-21 over `promotions.jsonl`: of 228 finished rows
# claiming `landed: true`, 23 name a round with no landing row at all and 6 more
# have only a `vault_land` while the round's surface was `code` or `mixed`.
# `SM_20260916_095854` (#415) is one of the 6 — "Landed in two halves: vault
# commit 5f65d971 … code commit b0818d4 …", and `main` never moved, which spent
# the item's single unattended attempt. The tests below are the #415 row's shape
# read out of that ledger, then the same claim through the writer.

LANDING_CLAIM = {"acceptance": "met", "landed": True, "clause_outcomes": [],
                 "deferred_to": [], "summary": "Landed in two halves", "spawned": []}
UNLANDED_ROUND = "SM_20260916_095854"


def _vault_row(item_id=415, *, ok=True):
    return {"event": "vault_land", "item_id": item_id, "ok": ok, "commit": "5f65d971" + "0" * 32}


def _reconcile(events, *, surface="mixed", round_id=UNLANDED_ROUND, item_id=415,
               landing_seen=False):
    """The self-report as the finalizer hands it over, reconciled once."""
    outcome = B.parse_outcome(LANDING_CLAIM)
    outcome, _ = B.settle_item_verdict(outcome, landing_seen=landing_seen)
    return B.reconcile_outcome_landing(outcome, round_id=round_id, item_id=item_id,
                                       events=list(events), landing_seen=landing_seen,
                                       surface=surface)


def test_a_mixed_round_with_only_a_vault_landing_is_recorded_as_not_landed():
    """The #415 case, exactly: a `vault_land` row and no promotion, `landed: true`.

    Accepting any landing row would pass this claim — the vault half really did
    commit, which is why the round's own summary reads true and why a mixed round
    is where the defect hides. On a `code`/`mixed` surface only `promoted` /
    `item_landed` says the diff reached `main`.
    """
    outcome, mismatch = _reconcile([_vault_row()])
    assert outcome["landed"] is False, "the half-landing stored the self-report verbatim"
    assert UNLANDED_ROUND in mismatch and "promoted" in mismatch, mismatch
    assert outcome["landed_mismatch"] == mismatch
    # The reconciliation demotes the flag, not the verdict: the round still said
    # `met`, and rewriting that here would move a judgment that is not this one.
    assert outcome["acceptance"] == "met"


@pytest.mark.parametrize("event", ["promoted", "item_landed"])
def test_a_round_with_a_code_landing_row_keeps_its_landed_claim(event):
    outcome, mismatch = _reconcile([{"event": event, "round_id": UNLANDED_ROUND,
                                     "item_id": 415, "commit": "a" * 40}])
    assert outcome["landed"] is True and mismatch == ""


def test_a_round_with_no_landing_row_at_all_is_recorded_as_not_landed():
    outcome, mismatch = _reconcile([{"event": "gate", "round_id": UNLANDED_ROUND,
                                     "rung": "review", "ok": True}], surface="code")
    assert outcome["landed"] is False and "promoted" in mismatch


def test_an_unsuccessful_vault_land_row_is_not_a_landing():
    """`ok: false` is a validation failure that reverted its own paths."""
    outcome, mismatch = _reconcile([_vault_row(ok=False)], surface="vault")
    assert outcome["landed"] is False and mismatch, "a reverted vault land counted as landed"


def test_a_vault_surface_round_lands_on_its_vault_row():
    """The other half of the surface rule: a vault-only item has no `promoted`
    row by construction, and demoting it would flag every vault round."""
    outcome, mismatch = _reconcile([_vault_row()], surface="vault")
    assert outcome["landed"] is True and mismatch == ""


def test_a_landing_still_in_flight_keeps_the_claim():
    """`automod_land` returns before the landing runs, so the ledger can be
    legitimately silent at the moment the turn is finalised. The process's own
    evidence — the live marker or `current.json` — covers that window."""
    outcome, mismatch = _reconcile([], landing_seen=True)
    assert outcome["landed"] is True and mismatch == ""


class _Payload:
    def __init__(self, payload):
        self.payload = payload


def _implement_turn_isolated(board, monkeypatch, structured, *, mid_turn=None,
                             surface=""):
    """Run one real implement turn against an isolated board and ledger, and
    return its `finished` ledger row. `mid_turn` fires inside the turn, which is
    when a landing's ledger row would actually appear; it may be async, for a hook
    that has to cross a tool handler (`agent_mcp/automod.py:call_tool`)."""
    import asyncio
    import inspect

    from workers.sources import _common as C

    write_item(board, 415)
    _confirm(415, acceptance="the check passes", surface=surface)

    async def fake(prompt, **kw):
        if mid_turn:
            hooked = mid_turn()
            if inspect.isawaitable(hooked):
                await hooked
        return {"text": "done\n\nSPAWNED: none\n", "session_id": "s415",
                "stop_reason": "stop", "num_turns": 30, "errors": [],
                "structured": structured, "structured_error": ""}

    async def no_reap(session_id):
        return None

    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    monkeypatch.setattr(I, "_reap_at_turn_end", no_reap)
    asyncio.run(I.execute(_Payload({"structured_outcome": True, "max_turns": 40})))
    return [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]


def test_a_vault_landing_row_is_not_evidence_that_the_round_promoted():
    """The two landing checks must not disagree about what a row means.

    `reconcile_outcome_landing` reads `promoted` / `item_landed` / `vault_land`
    (`backlog.LANDING_LEDGER_EVENTS`) and asks `_landing_seen` whether a landing was
    in flight, as the one exemption that covers `automod_land` returning before its
    landing runs. If `_landing_seen` also counted a `vault_land` row, every mixed
    half-landing would take that exemption and the reconciliation would never fire
    — the exact #415 shape, passing its own guard. The two lists are asserted
    against each other, through the real function, because the difference between
    them *is* the mixed-surface rule.
    """
    row = lambda event: {"event": event, "round_id": "SM_VOCAB", "item_id": 415,
                         "ok": True, "commit": "c" * 40}
    assert "vault_land" in B.LANDING_LEDGER_EVENTS, (
        "the reconciliation no longer recognises a vault landing at all, so a "
        "vault-surface round would be demoted on a landing it really did make")
    assert not I._landing_seen("SM_VOCAB", [row("vault_land")]), (
        "`_landing_seen` read a vault landing as this round reaching `automod_land`")
    assert I._landing_seen("SM_VOCAB", [row("promoted")])
    assert I._landing_seen("SM_VOCAB", [row("item_landed")])


def test_the_writer_demotes_an_unlanded_self_report_on_the_finished_row(isolated, monkeypatch):
    """The seam itself: the finalizer's structured object in, the persisted row
    out. #415's row said `"landed": true` beside a `round_aborted` four minutes
    later; a row that still says that after this change is the defect."""
    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM)
    assert ev["outcome"]["landed"] is False, ev["outcome"]
    assert "promoted" in ev["outcome_landing_mismatch"], ev["outcome_landing_mismatch"]
    assert ev["outcome"]["acceptance"] == "met", "the verdict is not what this reconciles"


def test_the_writer_demotes_a_mixed_round_whose_only_landing_is_the_vault(isolated, monkeypatch):
    """#415 end to end: the vault half committed, the code half did not, `landed: true`.

    The mixed-surface rule is only reachable if the in-flight escape hatch does
    not read a `vault_land` row as "the round reached `automod_land`" — that row
    is a landing of the other half. So this is the integration test for the
    wiring: the row appears inside the turn, the real writer runs, and the
    persisted row says not-landed.
    """
    def vault_half():
        S.append_event({"event": "vault_land", "item_id": 415, "ok": True,
                        "round_id": "SM_MIXED", "commit": "5f65d971" + "0" * 32},
                       path=S.LEDGER_PATH)

    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM,
                                  mid_turn=vault_half, surface="mixed")
    assert ev["outcome"]["landed"] is False, ev["outcome"]
    assert "promoted" in ev["outcome_landing_mismatch"], ev["outcome_landing_mismatch"]


def test_the_writer_keeps_a_claim_the_round_actually_promoted(isolated, monkeypatch):
    """The false-positive direction, same seam: a round that did land must not be
    demoted, or every landed item's row starts lying the other way."""
    def landed():
        S.append_event({"event": "round_start", "round_id": "SM_TEST_LANDED",
                        "item_id": 415, "session_id": "s415"}, path=S.LEDGER_PATH)
        S.append_event({"event": "promoted", "round_id": "SM_TEST_LANDED", "item_id": 415,
                        "commit": "b" * 40}, path=S.LEDGER_PATH)

    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM, mid_turn=landed)
    assert ev["outcome"]["landed"] is True, ev["outcome"]
    assert ev["outcome_landing_mismatch"] == ""


def test_a_landing_that_finished_after_the_snapshot_is_still_a_landing(tmp_path, monkeypatch):
    """The first seam the review rung named on `SM_20260921_091624`: the marker and
    `current.json` are read live, the ledger is a snapshot taken earlier in the
    run, so a landing that completes inside that window leaves both empty — the
    marker cleared, its `promoted` row appended after the snapshot — and the round
    that DID land gets demoted.

    The row is written with the real writer (`S.append_event`, what `promote.py`
    calls) into the real ledger file this test isolated, and the marker is absent
    because the landing is over. What is stale is the injected snapshot, which is
    exactly the condition under test. The mutation control is the same call with
    nothing in the ledger: the second look must not become an unconditional True.
    """
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "read_land_marker", lambda rid: None)
    monkeypatch.setattr(S, "read_current", lambda: {"round_id": "SM_OTHER"})
    S.append_event({"event": "promoted", "round_id": "SM_RACE", "commit": "e" * 40},
                   path=S.LEDGER_PATH)
    assert I._landing_seen("SM_RACE", []), (
        "a landing that completed after the run's snapshot was read as no landing — "
        "the round that landed is demoted to `landed: false`")
    # Control: same stale snapshot, same absent marker, but a ledger with no
    # landing row at all. The second look is a re-read of a file, not a yes.
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "empty-ledger.jsonl")
    assert not I._landing_seen("SM_RACE", [
        {"event": "round_start", "round_id": "SM_RACE"}]), (
        "`_landing_seen` answered True for a round with no landing row anywhere, "
        "which makes the second look meaningless")


def test_a_vault_landing_with_no_item_id_is_attributed_to_the_turn_that_made_it(
        isolated, monkeypatch, tmp_path):
    """The second seam the review rung named: `item_id` is OPTIONAL on
    `automod_vault_land`, and 56 of the 172 `vault_land` rows in the ledger on
    2026-09-21 carry none, so matching a vault landing on `item_id` alone demotes a
    vault-surface turn that really landed.

    End to end across the process boundary the row actually crosses: the row is
    written by the real MCP handler (`agent_mcp/automod.py`, a different process in
    production) with no item_id and no open round, into the ledger the worker
    reads; the fake implement turn reports the same session id the harness put in
    `_meta`. The claim must survive. The control is the same turn with no landing
    at all — attribution is not a licence.
    """
    import json

    import agent_mcp.automod as AM

    vault = tmp_path / "obsidian"
    (vault / "backlog").mkdir(parents=True)
    git(vault, "init", "-q", "-b", "main", str(vault))
    git(vault, "config", "user.email", "t@e.com")
    git(vault, "config", "user.name", "t")
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: draft\n---\n# item\n")
    git(vault, "add", "-A")
    git(vault, "commit", "-q", "-m", "base")
    # The change the tool is asked to land: the base commit must not contain it.
    (vault / "backlog" / "9-item.md").write_text("---\nstatus: done\n---\n# item\n")
    monkeypatch.setattr(V, "VAULT", vault)
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    monkeypatch.setattr(AM, "_inner_voice_gate", lambda action: None)

    async def land_without_an_item():
        from agent_mcp import _task_registry
        token = _task_registry.current_session_id.set("s415")
        try:
            # The real handler, awaited exactly as the MCP server awaits it.
            res = await AM.call_tool("automod_vault_land", {
                "paths": ["backlog/9-item.md"], "message": "backlog: promote #9"})
        finally:
            _task_registry.current_session_id.reset(token)
        out = json.loads(res[0].text if isinstance(res, list) else res.content[0].text)
        assert out.get("ok"), out
        return out

    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM,
                                  mid_turn=land_without_an_item, surface="vault")
    assert ev["outcome"]["landed"] is True, (
        f"a vault landing the handler really committed was attributed to nobody: {ev}")
    assert ev.get("outcome_landing_mismatch", "") == "", ev
    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "vault_land"]
    assert len(rows) == 1 and rows[0]["item_id"] is None, rows
    assert rows[0]["session_id"] == "s415", (
        "the row carries no attribution at all, so nothing downstream can tell a "
        "landing from a round that landed nothing")


def test_the_vault_writer_still_demotes_a_turn_that_landed_nothing(isolated, monkeypatch):
    """The control for the seam above, same surface and same claim: a `vault`
    turn whose ledger holds no `vault_land` row keeps being demoted. Attributing a
    row by session is not the same as trusting the claim."""
    ev = _implement_turn_isolated(isolated, monkeypatch, LANDING_CLAIM, surface="vault")
    assert ev["outcome"]["landed"] is False, ev["outcome"]
    assert "vault_land" in ev["outcome_landing_mismatch"], ev["outcome_landing_mismatch"]


def test_a_commit_the_vault_holds_is_not_an_unresolved_citation(repo, tmp_path, monkeypatch):
    """A `mixed` round lands vault commits and the grader cites them. Three of
    the five shas that spent review attempts on 2026-09-26/27 were real vault
    commits (7709cf49, e3c415a1, da4d1b42): the rail asked only the code repo."""
    r, _ = repo
    vault = tmp_path / "vault"
    vault.mkdir()
    git(vault, "init", "-q")
    (vault / "note.md").write_text("n\n")
    git(vault, "add", "-A")
    git(vault, "-c", "user.name=v", "-c", "user.email=v@v", "commit", "-q", "-m", "vault")
    vsha = git(vault, "rev-parse", "HEAD").stdout.strip()[:8]
    monkeypatch.setattr(RV, "REVIEW_EVIDENCE_ROOTS", (vault,))
    parsed = _judged(r, _fin({"verdict": "met", "test_node_id": "", "evidence_path": "app/m.py",
                              "evidence_line": 1, "note": f"the skill landed in vault commit {vsha}"}),
                     1, repo=r)
    assert parsed["unreliable"] == [] and "citation_unresolved" not in parsed["clauses"][0]
    assert RV.unresolved_shas(f"see {vsha}", r, also=()) == [vsha], "the code repo alone lacks it"


# ── the second reader on a block (#1903) ─────────────────────────────────
#
# A pass stands on one vote; a block is put to a second reader that can only
# demote it. These pin the rule that decides WHICH blocks are put to it, which
# entries it is shown, and what its answer may and may not do. The rung's own
# half — the ledger row, the refunded attempt — is in tests/test_automod_gate.py.

_CLAUSES_1903 = ["the block is confirmed before it costs an attempt",
                 "the confirming reader sees only the named finding"]


def _confirm_parsed(*, clauses=None, honesty=(), seams=(), premise="sound",
                    summary="SUMMARY_SENTINEL the first reader approved nothing else",
                    amendments_ok=True, amendments_note=""):
    """A parsed grader verdict, in the shape `_grade_entries` reads."""
    return {"premise": premise,
            "clauses": list(clauses or [{"clause": 1, "verdict": "unmet",
                                         "note": "ENTRY_ONE_SENTINEL no second reader exists"}]),
            "test_honesty": list(honesty), "seams_unverified": list(seams),
            "summary": summary, "downgraded": [],
            "amendments_ok": amendments_ok, "amendments_note": amendments_note}


def _honesty_entry(**kw):
    h = {"file": "tests/test_x.py", "line": 9, "problem": "`or True` makes it vacuous"}
    h.update(kw)
    return h


def test_the_confirm_policy_defaults_off_and_only_says_on_when_config_says_on(monkeypatch):
    """Off is the shipped setting, and off means the reader is never called.

    Two spellings are accepted, and every other value — including a config with
    no `automod.review.confirm` key at all, which is what every checkout has
    today — is off. A config that cannot be read is off too: this switch may
    add a grader turn to a refusal, it may never break the rung that reads it.
    """
    import app.config as C

    monkeypatch.setattr(C, "CONFIG", {"automod": {"review": {"seams_block": "never"}}})
    assert RV.confirm_policy() == RV.CONFIRM_OFF
    monkeypatch.setattr(C, "CONFIG", {"automod": {"review": {"confirm": "on"}}})
    assert RV.confirm_policy() == RV.CONFIRM_ON
    monkeypatch.setattr(C, "CONFIG", {"automod": {"review": {"confirm": True}}})
    assert RV.confirm_policy() == RV.CONFIRM_ON
    for off in (False, "off", "", None, "sometimes"):
        monkeypatch.setattr(C, "CONFIG", {"automod": {"review": {"confirm": off}}})
        assert RV.confirm_policy() == RV.CONFIRM_OFF, off
        assert RV.confirm_policy_value(off) is False, off

    class _Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("config unreadable")

    monkeypatch.setattr(C, "CONFIG", _Boom())
    assert RV.confirm_policy() == RV.CONFIRM_OFF, "a broken config never turns the reader on"


def test_shadow_is_its_own_confirm_state_and_nothing_else_resolves_to_it(monkeypatch):
    """#2017: three states, and `shadow` is neither neighbour's spelling.

    Before this, `confirm: shadow` in config.yaml silently meant OFF — the value
    was unrecognised and the policy was a bool — so a person who set it to start
    the measurement would have waited ten days for rows that were never written.
    The other direction matters as much: an unrecognised, absent or unreadable
    value must resolve to `off`, never to a state that calls a model on a block.
    """
    import app.config as C

    assert len({RV.CONFIRM_OFF, RV.CONFIRM_SHADOW, RV.CONFIRM_ON}) == 3
    for raw in ("shadow", "Shadow", " SHADOW "):
        monkeypatch.setattr(C, "CONFIG", {"automod": {"review": {"confirm": raw}}})
        assert RV.confirm_policy() == RV.CONFIRM_SHADOW, raw
        assert RV.confirm_policy_value(raw) is False, "shadow is not a switch-on"
    for raw in ("on", "true", "yes", "1", True):
        assert RV.confirm_policy_state(raw) == RV.CONFIRM_ON, raw
    for raw in ("shadows", "shadow-on", "dry", "observe", "warn", 2, 0, [], {}, None, "",
                False, "off", {"mode": "shadow"}, ["shadow"]):
        monkeypatch.setattr(C, "CONFIG", {"automod": {"review": {"confirm": raw}}})
        assert RV.confirm_policy() == RV.CONFIRM_OFF, raw
    for cfg in ({}, {"automod": None}, {"automod": {"review": None}},
                {"automod": {"review": {}}}):
        monkeypatch.setattr(C, "CONFIG", cfg)
        assert RV.confirm_policy() == RV.CONFIRM_OFF, cfg

    class _Boom(dict):
        def get(self, *a, **k):
            raise RuntimeError("config unreadable")

    monkeypatch.setattr(C, "CONFIG", _Boom())
    assert RV.confirm_policy() == RV.CONFIRM_OFF


def _shipped_confirm_line_and_block() -> str:
    """The tracked `config.yaml`'s `confirm:` line under `automod.review`, with
    the comment block directly above it, as one string.

    `tests/conftest.py` answers `off` for `confirm` for every node in the suite,
    "whatever config.yaml carries this week", so a node that wants the shipped
    state has to parse the tracked file itself. The flip #2305 makes is a value
    *and* the sentence that records why, so both are read here.
    """
    lines = (ROOT / "config.yaml").read_text(encoding="utf-8").splitlines()
    idxs = [i for i, ln in enumerate(lines) if ln.strip().startswith("confirm:")]
    assert len(idxs) == 1, f"`confirm:` appears {len(idxs)} times in config.yaml"
    i = idxs[0]
    j = i - 1
    while j >= 0 and lines[j].strip().startswith("#"):
        j -= 1
    return "\n".join(lines[j + 1:i + 1])


def test_the_shipped_config_resolves_the_confirm_switch_to_off():
    """#2305 clause 1: the shipped file is `off`, pushed through the same
    function `confirm_policy()` uses to decide whether to call a reader.

    The shadow measurement is over, so the switch goes back to the state that
    asks no reader and writes no `review_confirm*` field. This reads the tracked
    file rather than `CONFIG`, so an overlay in the environment cannot make it
    green, and resolves the raw value through `confirm_policy_state` rather than
    comparing strings, so it fails if the spelling ever stops resolving.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    confirm = cfg["automod"]["review"]["confirm"]
    assert RV.confirm_policy_state(confirm) == RV.CONFIRM_OFF, confirm


def test_the_shipped_confirm_key_survives_with_its_decision_in_the_comment():
    """#2305 clause 2: the switch is still there, and its comment names the two
    items that measured it, so the next reader finds a declined arm and not a
    deleted one.

    An absent key resolves to `off` too, which is exactly why presence is pinned
    separately: the point of leaving the key is to leave the ruling behind, and a
    reader who finds no key at all learns only that nobody ever looked.
    """
    block = _shipped_confirm_line_and_block()
    assert block.rstrip().endswith("confirm: off"), block.splitlines()[-1]
    assert "#1903" in block, "the second reader's own item"
    assert "#2017" in block, "the shadow pass that measured it"


def test_the_shipped_confirm_comment_carries_the_readout_and_no_open_instruction():
    """#2305 clause 3: the comment states what the shadow pass measured, and no
    longer orders the next reader to run the measurement.

    The figures are the ledger's own shadow rows: 307 of them from
    2026-10-01T17:04Z to 2026-10-06T18:44Z, of which 73 were askable — 23
    overturned (31.5 %), 50 upheld — and 234 never offered a reader. The pending
    instruction has to go with the measurement it asks for, because it is the one
    sentence in the file that would send a future session to re-derive a ruling
    owed-check already made.
    """
    block = _shipped_confirm_line_and_block()
    for want in ("overturned 23", "73 askable", "31.5", "upheld 50", "not_asked 234",
                 "2026-10-01T17:04Z", "2026-10-06T18:44Z"):
        assert want in block, want
    assert "then decide" not in block, "a pending instruction would outlive its measurement"
    assert ">= 30" not in block and "30 rows" not in block, "no re-measurement order ships"


def test_the_second_reader_is_offered_the_entries_the_decision_itself_produced():
    """The entries put to the reader are the entries the refusal is made of.

    Both blocking entries here are synthesized from the grader's CLAUSE VERDICTS
    — it filed no finding of its own — which is the case that would otherwise be
    answered `not_asked`: `clause 2 partial (downgraded: not graded)` is written
    by the decision rule when a `met` was thrown out for lack of evidence, and a
    refusal made only of those must still be put to a second reader.
    """
    parsed = _confirm_parsed(clauses=[
        {"clause": 1, "verdict": "unmet", "note": "no second reader exists"},
        {"clause": 2, "verdict": "partial", "downgraded": ["not graded"],
         "note": "the pin is not in this diff"},
    ])
    d = RV.decide_with_entries(parsed, [])
    assert d["kind"] == "retry"
    assert [e["kind"] for e in d["blocking"]] == ["clause", "clause"]
    assert [e["text"] for e in d["blocking"]] == [
        "clause 1 unmet: no second reader exists",
        "clause 2 partial (downgraded: not graded): the pin is not in this diff"]
    # Byte-for-byte the entries the refusal is made of: the join is the only
    # other place they are written, so nothing can drift between the two.
    assert d["findings"] == RV.decide_by_grader(parsed, [])[1]
    for e in d["blocking"]:
        assert e["text"] in d["findings"]
    plan = RV.confirm_plan(d["kind"], d["blocking"], confirm_on=True)
    assert plan["ask"] is True and plan["reason"] == "ask"
    assert [e["text"] for e in plan["entries"]] == [e["text"] for e in d["blocking"]]


def test_a_python_computed_or_contract_refusal_is_never_put_to_the_reader():
    """Every exemption names its reason on the record, and none is a judgment.

    A Python pattern found a test-honesty fact; an `unsatisfiable` clause and a
    refused amendment are defects of the CONTRACT, not of this diff; `unsound`
    is a call about the ITEM. A reader aimed at "is this finding real about the
    diff" cannot un-find a pattern, un-refuse an amendment or re-judge the item,
    so none of these is asked — and each refusal says which rule spared it.
    """
    def plan_of(parsed, pre=(), amendments=None, **kw):
        d = RV.decide_with_entries(parsed, pre, amendments=amendments, **kw)
        return RV.confirm_plan(d["kind"], d["blocking"], confirm_on=True), d

    cases = {}
    # A blocking entry a Python pattern found, not a grader judgment.
    cases["all_python_computed"] = plan_of(
        _confirm_parsed(clauses=[{"clause": 1, "verdict": "met", "note": ""}]),
        pre=[_honesty_entry()])
    # Same fact from the grader's own list, at blocking severity and fixable.
    cases["all_python_computed_grader_filed"] = plan_of(
        _confirm_parsed(clauses=[{"clause": 1, "verdict": "met", "note": ""}],
                        honesty=[_honesty_entry(severity="blocking",
                                                actionable_in_round=True,
                                                testable_before_landing=True)]))
    cases["unsatisfiable_clause"] = plan_of(_confirm_parsed(clauses=[
        {"clause": 1, "verdict": "unsatisfiable", "note": "no diff can satisfy this"}]))
    cases["amendment_refused"] = plan_of(
        _confirm_parsed(clauses=[{"clause": 1, "verdict": "unmet", "note": "n"}],
                        amendments_ok=False, amendments_note="the clause is restored"),
        amendments=[{"clause": 1, "round_id": "SM_X", "text": "weaker"}])
    cases["premise_unsound"] = plan_of(_confirm_parsed(premise="unsound"))
    # A block with no clause verdict in it is not the thing this reader is for.
    cases["no_clause_verdict"] = plan_of(
        _confirm_parsed(clauses=[{"clause": 1, "verdict": "met", "note": ""}],
                        seams=[{"seam": "the loopback POST"}]), policy="always")
    # A refusal that is nothing but advisories never reaches a second reader.
    cases["not_a_refusal"] = plan_of(_confirm_parsed(
        clauses=[{"clause": 1, "verdict": "post_landing", "note": "wait for landing"}]))
    # The policy itself, off: the reader is not called and nothing is offered.
    cases["policy_off"] = (RV.confirm_plan("retry", [{"text": "clause 1 unmet: x",
                                                      "kind": "clause"}], confirm_on=False), None)

    for expected, (plan, decision) in cases.items():
        want = expected.split("_grader")[0]
        assert plan["ask"] is False, f"{expected}: must not be put to a reader"
        assert plan["entries"] == [], expected
        assert plan["reason"] == want, f"{expected}: reason was {plan['reason']}"
        assert plan["reason"] in RV.NOT_ASK_REASONS, plan["reason"]
    assert "test honesty tests/test_x.py:9" in cases["all_python_computed"][1]["findings"]
    assert "clause 1 unsatisfiable as written" in cases["unsatisfiable_clause"][1]["findings"]
    assert "amendment of clause(s) 1 refused" in cases["amendment_refused"][1]["findings"]
    # The positive control beside these negatives is the one entry that IS put
    # to a reader: a refusal built of nothing but an unrecognised entry shape
    # is not asked about either, which proves the classifier is reading the
    # entries and not just failing open.
    odd = RV.decide_with_entries(_confirm_parsed(clauses=[
        {"clause": 1, "verdict": "met", "note": ""}]), [])
    odd["blocking"] = [{"text": "something this rule has no spelling for", "kind": "other"}]
    assert RV.confirm_plan("retry", odd["blocking"], confirm_on=True)["reason"] == "no_clause_verdict"
    assert RV.confirm_plan("retry", [{"text": "clause 1 unmet: x", "kind": "clause"}],
                           confirm_on=True)["ask"] is True, "the classifier is not failing closed"


def test_the_reader_sees_the_diff_and_the_one_named_entry_and_nothing_of_the_first_reading():
    """What the second reader is shown, asserted against unique sentinels.

    Two blocking entries, asked about one at a time: the prompt for entry 1
    carries entry 1 and the diff, and must not carry entry 2's text, nor the
    first reading's summary. Both absences are load-bearing — a reader shown
    the first reader's conclusions is being primed, not consulted, and a reader
    shown a second finding can retire the wrong one.
    """
    parsed = _confirm_parsed(clauses=[
        {"clause": 1, "verdict": "unmet", "note": "ENTRY_ONE_SENTINEL no reader is wired"},
        {"clause": 2, "verdict": "unmet", "note": "ENTRY_TWO_SENTINEL the other finding"},
    ])
    d = RV.decide_with_entries(parsed, [])
    plan = RV.confirm_plan(d["kind"], d["blocking"], confirm_on=True)
    sent: list[dict] = []

    def fake_run(**kw):
        sent.append(kw)
        return {"ok": True, "error": "", "session_id": "sess_confirm",
                "structured": {"retire": True, "reason": "checked the diff, the pin is there"}}

    reader = RV.confirm_reader(round_id="SM_1903", item_id=1903, clauses=_CLAUSES_1903,
                              diff="DIFF_SENTINEL +def confirm_refusal():", run_grader_fn=fake_run)
    got = RV.confirm_refusal(d, plan, reader=reader)

    assert got["outcome"] == RV.OVERTURNED and got["asked"] == 2
    assert len(sent) == 2, "one turn per blocking entry, not one per diff"
    first = sent[0]["prompt"]
    assert "ENTRY_ONE_SENTINEL" in first and "DIFF_SENTINEL" in first
    assert "the block is confirmed before it costs an attempt" in first, "the clause it must judge against"
    assert "ENTRY_TWO_SENTINEL" not in first, "entry 2 is a different question"
    for kw in sent:
        assert "SUMMARY_SENTINEL" not in kw["prompt"], "the first reading's summary is not evidence"
        assert kw["final_schema"] is RV.CONFIRM_SCHEMA
        assert kw["round_id"] == "SM_1903" and kw["item_id"] == 1903


def test_a_second_reader_may_retire_only_the_entry_it_was_named():
    """Demote-only, in both directions, and fail-closed on every bad answer.

    The reader is asked about ONE entry, so its answer is attached to that entry
    and no other: it cannot retire a finding it was not shown, cannot retire one
    that a Python pattern found, and cannot add a finding (there is nowhere for
    one to go). Anything short of a JSON `true` upholds, so a confused or broken
    second reader costs a grader turn and changes nothing.
    """
    parsed = _confirm_parsed(clauses=[
        {"clause": 1, "verdict": "unmet", "note": "first"},
        {"clause": 2, "verdict": "unmet", "note": "second"},
    ])
    d = RV.decide_with_entries(parsed, [])
    plan = RV.confirm_plan(d["kind"], d["blocking"], confirm_on=True)
    asked: list[str] = []

    def retire_first_only(text):
        asked.append(text)
        return {"retire": text == d["blocking"][0]["text"],
                "reason": "the pin is on the line the note names",
                "new_blocking_finding": "clause 9 unmet: invented"}

    got = RV.confirm_refusal(d, plan, reader=retire_first_only)
    assert asked == [e["text"] for e in d["blocking"]], "asked about exactly the blocking entries"
    assert got["outcome"] == RV.UPHELD, "entry 2 stands, so the refusal stands"
    assert [v["verdict"] for v in got["votes"]] == ["retired", "upheld"]
    assert got["reason"] == ("clause retired: the pin is on the line the note names; "
                            "clause upheld: the pin is on the line the note names")
    assert len(got["votes"]) == 2, "an invented finding in the answer is recorded nowhere"
    assert "invented" not in got["reason"]

    # An entry no Python pattern found can be retired; one can never be asked
    # about, so it can never be retired even by a reader that would.
    mixed = RV.decide_with_entries(
        _confirm_parsed(clauses=[{"clause": 1, "verdict": "unmet", "note": "first"}]),
        [_honesty_entry()])
    got2 = RV.confirm_refusal(mixed, RV.confirm_plan("retry", mixed["blocking"], confirm_on=True),
                             reader=lambda t: {"retire": True, "reason": "everything is fine"})
    assert got2["outcome"] == RV.UPHELD and got2["asked"] == 1
    assert [v["entry"] for v in got2["votes"]] == ["clause 1 unmet: first"], \
        "the honesty entry is never offered, so it can never be retired"
    assert "test honesty" in mixed["findings"], "the honesty finding is still what refuses"

    for bad in ({"retire": "true", "reason": "a string is not a true"},
               {"retire": 1, "reason": "an int is not a true"},
               {"reason": "no retire key at all"},
               {"retire": False, "reason": "the finding stands"},
               "not even a dict", {}):
        got3 = RV.confirm_refusal(d, plan, reader=lambda t, b=bad: b)
        assert got3["outcome"] == RV.UPHELD, bad

    def boom(text):
        raise RuntimeError("the reader died")

    got4 = RV.confirm_refusal(d, plan, reader=boom)
    assert got4["outcome"] == RV.UPHELD and "RuntimeError" in got4["reason"]

    # Past the cap an entry is never asked about, so it cannot be overturned.
    many = RV.decide_with_entries(_confirm_parsed(clauses=[
        {"clause": n, "verdict": "unmet", "note": f"note {n}"}
        for n in range(1, RV.CONFIRM_MAX_ENTRIES + 2)]), [])
    got5 = RV.confirm_refusal(many, RV.confirm_plan("retry", many["blocking"], confirm_on=True),
                             reader=lambda t: {"retire": True, "reason": "all clear"})
    assert got5["asked"] == RV.CONFIRM_MAX_ENTRIES, "the offered entries are capped"
    assert got5["outcome"] == RV.UPHELD, "an entry never asked about is never retired"


# ── #2040: a deletion clause's evidence, on the vault shape ──────────────────
#
# `grade_vault` passed `parse_review` no `changed_paths`, so of the two readings
# `evidence_of_absence` accepts only the `(absent)` marker arm could fire here, and the
# vault prompt never told the grader that arm existed. #2038's two refusals are the
# outcome; the rows are committed at
# tests/fixtures/promotions_vault_review_rows_2026-10-01-item2038.jsonl and pinned by
# tests/test_automod_vault_round.py::test_the_promotions_witness_rows_still_show_the_rail_not_the_diff_refusing_them.

#: A leaf the lander listed and its own edit deleted — clause 1 of #2038's grading
#: cited exactly this, verbatim from the ledger row.
REMOVED_LEAF = ("_pipeline/tmp/pt1018/test_run_records_age_by_frontm0"
                "/autonomy-runs/24/run_24_20260903_120000.md")


def _vault_met(wt, monkeypatch, *, changed_paths=(), **clause):
    """A `met` graded the way a vault round grades one: `require_tests=False`, so no
    test node is asked for and the path rail is the only rail standing between the
    grader's word and a `partial`.

    `REVIEW_EVIDENCE_ROOTS` is emptied first: its one entry is the live vault, where
    `_pipeline/` still stands (attempt 2's 64 staged deletions were reverted), so an
    unisolated node here would be graded against the machine rather than the tree under
    review — the leak that made this node pass or fail on the state of `~/obsidian`.
    """
    monkeypatch.setattr(RV, "REVIEW_EVIDENCE_ROOTS", ())
    base = {"evidence_path": REMOVED_LEAF, "evidence_line": 0, "test_node_id": "",
            "how_verified": "read", "note": "witness is the path's absence"}
    base.update(clause)
    parsed = RV.parse_review(_obj(**base), worktree=wt, changed_tests=[], n_clauses=1,
                             require_tests=False, changed_paths=list(changed_paths))
    return parsed["clauses"][0]


def test_a_vault_met_citing_an_unlisted_unmarked_removal_is_still_downgraded(wt,
                                                                            monkeypatch):
    """The absence rail narrowed, not deleted — #2040 clause 2.

    Three arms of one grader object, one call each. A path the lander listed and removed
    is admitted, because the diff is the only witness a removal has. A path marked with an
    admissible marker is admitted, because that is what the marker is for. A path that is
    in neither — this file, on this surface, before the fix, for every clause — is refused
    with the words the ledger shows, and `accepted` stays empty: a waived rail announces
    itself, and a rail that fired does not.
    """
    listed = _vault_met(wt, monkeypatch, changed_paths=[".gitignore", REMOVED_LEAF])
    assert listed["verdict"] == "met" and "downgraded" not in listed
    assert "evidence of absence" in listed["accepted"][0], listed

    marked = _vault_met(wt, monkeypatch, evidence_path="_pipeline (absent)")
    assert marked["verdict"] == "met" and "evidence of absence" in marked["accepted"][0]

    neither = _vault_met(wt, monkeypatch, evidence_path="_pipeline")
    assert neither["verdict"] == "partial", "the rail stopped refusing anything"
    assert "evidence_path missing or not on disk" in neither["downgraded"][0]
    assert "'_pipeline'" in neither["downgraded"][0], "the refusal names what was cited"
    # `parse_review` clears the key and then omits it entirely (review.py's
    # `if accepted: out["accepted"] = accepted`), so the ONLY shape a refused clause
    # has is the absent key. Asserted as such: an `or not ...` half here would be a
    # branch no code path can reach, and a test arm that cannot fail is not a check.
    assert "accepted" not in neither, neither


def test_the_vault_prompt_offers_the_reviewer_a_marker_the_rails_accept(isolated):
    """#2040 clause 3: the prompt has to name a shape the rails will take.

    The code-surface prompt already names the escape; the vault one stopped at "`met`
    needs an evidence_path under the vault", so a grader grading a removal had a correct
    citation to write and no admissible way to write it. The marker asserted here is not a
    string this test picked: it is drawn from `RV._ABSENCE_MARKERS`, the same tuple
    `evidence_of_absence` matches, so a prompt that drifts away from what the rails accept
    — in either direction, wording or vocabulary — goes red here rather than in a round's
    last attempt.
    """
    p = RV.build_vault_prompt(contract=_vault_contract(["content"]),
                              paths=["skills/x/SKILL.md"], diff="+x", vault=isolated)
    marker = next(m for m in RV._ABSENCE_MARKERS if m in p)
    anchor = p.index("`met` needs an evidence_path under the vault")
    assert p.index(marker) > anchor, "the marker belongs beside the evidence rule"
    removed = [w for w in ("removed", "deletes", "delete", "gone") if w in p[anchor:]]
    assert removed, "the sentence has to say when the marker is the right citation"


# ── #2317: a moved constant, and the autonomy prose left quoting the old one ─
#
# The other write edge of the same drift. `01dea8bc` moved
# `LEDGER_ARCHIVE_AGE_DAYS` from 30 to 14 in `scripts/groundskeeper/retention-sweep.py`
# and nothing said that `autonomy/79-retention-sweep.md` was now wrong — the vault
# probe could not (no node read the description's numbers), and no rung on the code
# side looked either. These nodes pin the advisory that names it at the moment of the
# move. The land-side refusal is `test_automod_vault_round.py`'s three; this family is
# deliberately advisory, for the reason spelled out on `_CONSTANT_STALE_SEVERITY`.

#: The two shapes of prose the move is tested against: one file whose description
#: quotes the value this round leaves behind, one that has already been updated.
STALE_TASK = """---
id: 79
status: up_next
description: Sweep the ledger; LEDGER_ARCHIVE_AGE_DAYS (30) is the window.
---
# Retention sweep
"""

FRESH_TASK = """---
id: 76
status: up_next
description: Queue age over QUEUE_AGE_HOURS (6) is escalated.
---
# Queue health
"""


@pytest.fixture
def move_repo(tmp_path):
    """A round whose only change is one module constant, 30 -> 14."""
    r = tmp_path / "r"
    (r / "scripts").mkdir(parents=True)
    git(tmp_path, "init", "-q", "-b", "main", str(r))
    git(r, "config", "user.email", "t@e.com"); git(r, "config", "user.name", "t")
    (r / "scripts" / "retention-sweep.py").write_text("LEDGER_ARCHIVE_AGE_DAYS = 30\n")
    git(r, "add", "-A"); git(r, "commit", "-q", "-m", "base")
    base = git(r, "rev-parse", "HEAD").stdout.strip()
    v = tmp_path / "obsidian"
    (v / "autonomy").mkdir(parents=True)
    (v / "autonomy" / "79-retention-sweep.md").write_text(STALE_TASK)
    (v / "autonomy" / "76-queue-health.md").write_text(FRESH_TASK)
    return r, base, v


def _move_the_constant(r: Path) -> None:
    (r / "scripts" / "retention-sweep.py").write_text("LEDGER_ARCHIVE_AGE_DAYS = 14\n")
    git(r, "commit", "-qam", "shorten the ledger window")


def test_a_moved_constant_names_the_description_quoting_the_old_value(move_repo):
    """Clause 4's finding: the file, the constant, and BOTH numbers.

    `76-queue-health.md` quotes a different constant that this round did not move, so
    it must not be named: a finding list padded with files that are fine is how a
    advisory becomes noise the next author skims past.
    """
    r, base, v = move_repo
    _move_the_constant(r)
    out = RV.stale_constant_quotes(r, base, ["scripts/retention-sweep.py"], vault=v)
    assert len(out) == 1, out
    assert out[0]["file"] == "autonomy/79-retention-sweep.md"
    problem = out[0]["problem"]
    assert "LEDGER_ARCHIVE_AGE_DAYS" in problem and "30" in problem and "14" in problem
    assert out[0]["severity"] == "advisory"


def test_a_move_that_leaves_no_description_stale_makes_no_finding(move_repo):
    """Green half: prose already naming the new value is not a finding.

    Without this the check would be indistinguishable from one that names every
    autonomy file on any code change, which is the shape that gets skimmed.
    """
    r, base, v = move_repo
    (v / "autonomy" / "79-retention-sweep.md").write_text(
        STALE_TASK.replace("(30)", "(14)"))
    _move_the_constant(r)
    assert RV.stale_constant_quotes(r, base, ["scripts/retention-sweep.py"], vault=v) == []


def test_a_constant_this_round_did_not_move_is_not_reported(move_repo):
    """The delta, from the other side: the same stale prose, and no finding.

    Nothing in this diff moved the constant, so this is a pre-existing disagreement
    between the vault and the tree — the vault edge's business (`autonomy_description_errors`
    refuses a land that re-publishes it), not a claim about this round. A precheck that
    reported the tree's general state would name every round for a drift no round
    caused, which is the mistake #1019 records for the honesty prechecks.
    """
    r, base, v = move_repo
    (r / "scripts" / "retention-sweep.py").write_text(
        "# comment only; the constant did not move\nLEDGER_ARCHIVE_AGE_DAYS = 30\n")
    git(r, "commit", "-qam", "comment")
    assert RV.stale_constant_quotes(r, base, ["scripts/retention-sweep.py"], vault=v) == []
    assert RV.moved_int_constants(r, base, ["scripts/retention-sweep.py"]) == []


def test_a_non_integer_change_is_not_a_move(move_repo):
    """`NAME = <int>` is the only spelling a description can quote, so it is the only
    one that counts: a `str` or a formula has no number for prose to disagree with.
    """
    r, base, v = move_repo
    (r / "scripts" / "retention-sweep.py").write_text(
        "LEDGER_ARCHIVE_AGE_DAYS = 14 * 2\n")
    git(r, "commit", "-qam", "derived, not literal")
    assert RV.moved_int_constants(r, base, ["scripts/retention-sweep.py"]) == []
    assert RV.stale_constant_quotes(r, base, ["scripts/retention-sweep.py"], vault=v) == []


def test_the_stale_quote_finding_rides_the_prechecks_the_gate_records(move_repo):
    """The wiring, through the real funnel rather than a stub of it.

    `honesty_prechecks` is the list the gate writes onto the review event
    (`scripts/automod/gate.py`) and `review_tools.grade_commit` re-uses for historical
    commits, so a finding that only `stale_constant_quotes` produces, and that nothing
    appends, would exist in no report at all.
    """
    r, base, v = move_repo
    _move_the_constant(r)
    out = RV.honesty_prechecks(r, base, ["scripts/retention-sweep.py"], vault=v)
    hits = [o for o in out if "LEDGER_ARCHIVE_AGE_DAYS" in o["problem"]]
    assert len(hits) == 1, out
    assert hits[0]["severity"] == "advisory"


def test_an_advisory_stale_quote_never_refuses_the_round(move_repo):
    """Clause 4's other half, checked at the one place severity becomes a decision.

    `_grade_entries` splits prechecks on `severity == "advisory"` alone, so this is
    not prose about the design — it is the branch. With a clean grader object the
    round must PASS with the finding in the returned text, not retry: the fix is a
    vault edit the round's own diff cannot contain.
    """
    r, base, v = move_repo
    _move_the_constant(r)
    pre = RV.honesty_prechecks(r, base, ["scripts/retention-sweep.py"], vault=v)
    assert pre, "the fixture stopped producing the finding this decides on"
    parsed = {"premise": "sound", "summary": "ok", "test_honesty": [],
              "seams_unverified": [], "downgraded": [],
              "clauses": [{"clause": 1, "verdict": "met", "note": ""}]}
    kind, text = RV.decide(parsed, pre, policy="never")
    assert kind == "pass", (kind, text)
    assert "advisory autonomy/79-retention-sweep.md" in text, text


def test_the_direction_of_the_reported_move_is_pinned(move_repo):
    """`(name, file, line, old, new)` in that order, with the real move's numbers.

    The finding's sentence is "was X, now Y", and the review rung's own `_mirror_gap`
    history shows how much a swapped pair costs: the mirror detector above unpacked
    this same pair under the name `new` for the BASE value, which is how a reversed
    (old, new) can reach a finding text while every count still looks right. A node
    that asserted only `len(out) == 1` and the name would pass that inversion, so
    this one asserts the two numbers by position — 30 was the tree's window before
    `01dea8bc`, 14 after, in that order.
    """
    r, base, v = move_repo
    _move_the_constant(r)
    out = RV.moved_int_constants(r, base, ["scripts/retention-sweep.py"])
    assert len(out) == 1, out
    name, path, line, old, new = out[0]
    assert (name, str(path)) == ("LEDGER_ARCHIVE_AGE_DAYS", "scripts/retention-sweep.py")
    assert (old, new) == (30, 14), f"reported the move backwards: was {new}, now {old}"
    assert line == 1
    # And the finding sentence carries the pair in the same order the tuple does, so a
    # reader can tell which of the two numbers the description is stale against.
    problem = RV.stale_constant_quotes(r, base, ["scripts/retention-sweep.py"],
                                       vault=v)[0]["problem"]
    assert f"as {old};" in problem and f"moves it {old} -> {new}" in problem, problem


# ── #2263: a grading is bounded per call, and no grading is a failure ────────
#
# #2240 capped every string leaf the review schema admits on the theory that an
# uncapped `note` was what let a grading run past `harness.finalizer.max_tokens`
# (8192, config.yaml:744). The theory was right about the earlier failures and
# wrong about the ceiling: three `vault_review` rows diverged AFTER those caps
# went live on 2026-10-05T17:55Z — item 2260 at 2026-10-06T01:31:39Z, 2287 at
# 13:39:20Z, 2325 at 2026-10-07T02:59:59Z — all `kind: skipped`, `clauses: []`,
# `blocking: false`. Every one is a 5- or 6-clause contract and no 4-clause
# review diverged, so the bound has to sit on the CALL. A cap of 3 is a structural
# property, not a flaky threshold: `CLAUSES_PER_CALL` is what the tests below
# measure against, and the census is `grep -c 'generation diverged'
# ~/.local/state/lloyd-automod/promotions.jsonl` = 21 as this was written.

SIX_CLAUSES = [f"clause {i} of six is satisfied on disk" for i in range(1, 7)]


def _met_answer(indices):
    """A grader answer with one `met` row for each of `indices`, evidence real."""
    return {"ok": True, "structured": {
        "premise": "sound", "summary": "each clause read off disk", "test_honesty": [],
        "seams_unverified": [],
        "clauses": [{"clause": i, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
                     "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                     "note": f"read line 1 for clause {i}"} for i in indices]}}


def _six_clause_item(isolated, item_id, tmp_path):
    write_item(isolated, item_id, clauses=SIX_CLAUSES)
    _confirm(item_id, acceptance="; ".join(SIX_CLAUSES), clauses=SIX_CLAUSES, surface="vault")
    (isolated / "skills" / "x").mkdir(parents=True, exist_ok=True)
    (isolated / "skills" / "x" / "SKILL.md").write_text("x\n", encoding="utf-8")


def _asked_clauses(prompt):
    """Which of the six clauses THIS prompt names, by its rendering in
    `<acceptance_clauses>`. Detecting what was asked rather than trusting the
    caller's chunk is what makes the next test a measurement."""
    return [i for i in range(1, 7) if f"{i}. {SIX_CLAUSES[i - 1]}" in prompt]


def test_clause_chunks_cover_the_contract_in_calls_of_three():
    """The structural bound: `ceil(n/3)` calls, every index 1..n in exactly one."""
    assert RV.CLAUSES_PER_CALL <= 3, "the bound #2263 pins is at most three clauses a call"
    assert RV.clause_chunks(6) == [[1, 2, 3], [4, 5, 6]]
    assert RV.clause_chunks(7) == [[1, 2, 3], [4, 5, 6], [7]]
    assert RV.clause_chunks(3) == [[1, 2, 3]] and RV.clause_chunks(4) == [[1, 2, 3], [4]]
    assert RV.clause_chunks(0) == [] and RV.clause_chunks(None) == []
    for n in range(1, 21):
        chunks = RV.clause_chunks(n)
        flat = [i for chunk in chunks for i in chunk]
        assert flat == list(range(1, n + 1)), f"{n}: gap or duplicate in {chunks}"
        assert len(chunks) == -(-n // RV.CLAUSES_PER_CALL)
        assert all(len(c) <= RV.CLAUSES_PER_CALL for c in chunks)


def test_a_six_clause_vault_contract_is_graded_in_two_calls_of_three_clauses(
        isolated, monkeypatch, tmp_path):
    """#2260's contract, end to end through `grade_vault` with only the network
    seam replaced. One generation over six clauses is what overran 8192 tokens at
    2026-10-06T01:31:39Z and left `ea9be106` with no verdict; six clauses are the
    common case, not the tail."""
    _six_clause_item(isolated, 601, tmp_path)
    seen = []

    def fake(**kw):
        asked = _asked_clauses(kw["prompt"])
        seen.append(asked)
        return _met_answer(asked)
    monkeypatch.setattr(RV, "run_grader", fake)
    kind, why, clauses = RV.grade_vault(item_id=601, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert seen == [[1, 2, 3], [4, 5, 6]], f"one call per slice, in contract order: {seen}"
    assert kind == "pass", why
    assert [c["clause"] for c in clauses] == [1, 2, 3, 4, 5, 6]
    assert all(c["verdict"] == "met" for c in clauses), clauses


def test_a_grading_call_names_only_the_clauses_it_is_answering(isolated, monkeypatch, tmp_path):
    """The chunk keeps the CONTRACT's numbering, so the merged rows need no
    translation and a grader cannot renumber its way into a gap."""
    _six_clause_item(isolated, 602, tmp_path)
    prompts = []

    def answer_what_is_asked(**kw):
        prompts.append(kw["prompt"])
        return _met_answer(_asked_clauses(kw["prompt"]))
    monkeypatch.setattr(RV, "run_grader", answer_what_is_asked)
    RV.grade_vault(item_id=602, paths=["skills/x/SKILL.md"], diff="+x", vault=isolated,
                   attempt=1)
    # Nothing fails here, so the two PLANNED calls are the whole transcript (#2341
    # added a re-ask, and a re-ask needs a failure to be about), and the second is
    # the one that asks for 4-6.
    assert len(prompts) == 2, "a grading that fails nothing issues only planned calls"
    last = prompts[-1]
    for i in (4, 5, 6):
        assert f"{i}. {SIX_CLAUSES[i - 1]}" in last
    for i in (1, 2, 3):
        assert f"{i}. {SIX_CLAUSES[i - 1]}" not in last, "clause 1 is not this call to grade"
    assert "Grade ONLY clauses 4, 5, 6" in last
    assert "already the item's own; do not renumber" in last
    # The whole-contract call asks for clauses 1-3 and says nothing about 4-6's
    # texts, so nothing outside the slice can be answered from it.
    first = prompts[0]
    assert "Grade ONLY clauses 1, 2, 3" in first
    assert SIX_CLAUSES[5] not in first
    # A contract that fits one call is prompted exactly as before #2263.
    single = RV.build_vault_prompt(contract=_vault_contract(["only clause"]),
                                   paths=["a.md"], diff="+a", vault=isolated,
                                   clause_indices=[1])
    assert "<which_clauses_to_grade>" not in single, "no batch note when nothing is batched"


def test_a_merged_grading_that_leaves_a_clause_unanswered_is_no_verdict(
        isolated, monkeypatch, tmp_path):
    """#2263 clause 2. A grader that answers 4 of 6 did not grade the contract;
    before this it arrived as a `retry` refusal carrying a synthesized `partial`
    for the clause nobody looked at, which is a verdict on the change invented
    from the grader's silence.

    The stub now answers every generation the same way — clause 5 is the one it
    cannot answer, at any size — because #2341 clause 1 made `grade_vault` re-ask
    the skipped index at one clause per call before it may report anything. The
    assertion this test exists for is unchanged: a contract with an unanswerable
    clause comes back `incomplete` with NO verdicts, and the refusal now names the
    shrink it tried first."""
    _six_clause_item(isolated, 603, tmp_path)
    asked = []

    def never_clause_5(**kw):
        slice_ = _asked_clauses(kw["prompt"])
        asked.append(slice_)
        return _met_answer([i for i in slice_ if i != 5])
    monkeypatch.setattr(RV, "run_grader", never_clause_5)
    kind, why, clauses = RV.grade_vault(item_id=603, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind == RV.GRADER_INCOMPLETE
    assert clauses == [], "an unanswered clause must not be answered for the grader"
    assert "no verdict for clause(s) 5" in why, why
    # The skipped index was re-asked alone, not merely reported: 1-3, then 4-6,
    # then clause 5 by itself at the smallest chunk there is.
    assert asked == [[1, 2, 3], [4, 5, 6], [5]], asked
    assert "shrank to 1 clause per call" in why, why


def _answer(rows) -> dict:
    """One grader answer. `rows` is `[(clause_index, verdict, note), …]`, and an index may
    appear twice — which is the shape a clause with two subjects produces (#2442)."""
    return {"ok": True, "structured": {
        "premise": "sound", "summary": "read off disk", "test_honesty": [],
        "seams_unverified": [],
        "clauses": [{"clause": c, "verdict": v, "evidence_path": "skills/x/SKILL.md",
                     "evidence_line": 1, "test_node_id": "", "how_verified": "read",
                     "note": n} for c, v, n in rows]}}


def test_a_clause_graded_twice_merges_to_one_verdict_and_the_contract_grades(isolated,
                                                                             monkeypatch):
    """#2442 clause 1: a repeat is a conjunction now, not `kind: incomplete`.

    Item #2434's clause 5 asks for one figure in BOTH runbooks, so the grader answers
    clause 5 twice — once per changed file — and `merge_grading_chunks` refused the whole
    contract `incomplete`, which `vault_round.land` charged `blocking: true` on the FIRST
    attempt and answered by reverting `skills/nightly-skill-consolidation/SKILL.md`, the
    file whose edit the same grader's findings had just called correct. Four other items
    (#2335, #2372, #2410, #2411) died the same way on ONE-path contracts, so this is not
    about how many files a round passes.

    Here the same six-clause contract — the one whose death this file used to pin — is
    answered with clause 5 graded twice, and the run must produce a real verdict instead
    of `RV.GRADER_INCOMPLETE`, with ONE clause-5 row carrying the worst of the two answers
    and both answers underneath it.
    """
    _six_clause_item(isolated, 604, tmp_path=isolated)
    def dup(**kw):
        slice_ = _asked_clauses(kw["prompt"])
        if slice_ == [1, 2, 3]:
            return _met_answer([1, 2, 3])
        if slice_ == [4, 5, 6] or slice_ == [5]:
            return _answer([(4, "met", "consolidation figure present"),
                            (5, "met", "the management runbook carries it"),
                            (5, "unmet", "the consolidation runbook does not"),
                            (6, "met", "no stale figure")])
        return _answer([(i, "met", "graded") for i in slice_])
    monkeypatch.setattr(RV, "run_grader", dup)
    kind, why, clauses = RV.grade_vault(item_id=604,
                                       paths=["skills/nightly-skill-consolidation/SKILL.md",
                                              "skills/nightly-skills-management/SKILL.md"],
                                       diff="+x", vault=isolated)
    assert kind != RV.GRADER_INCOMPLETE, f"a duplicate must not be a grader fault: {why}"
    assert kind == "retry", (kind, why)
    fives = [c for c in clauses if c["clause"] == 5]
    assert len(fives) == 1, f"one clause index, ONE verdict: {clauses}"
    assert fives[0]["verdict"] == "unmet", fives[0]
    assert [v["verdict"] for v in fives[0]["sub_verdicts"]] == ["met", "unmet"], fives[0]
    assert len([c for c in clauses if c["clause"] == 4]) == 1, clauses


def test_the_merged_clause_verdict_does_not_depend_on_which_answer_came_first(monkeypatch):
    """#2442 clause 2: worst-wins, so `met` cannot hide `unmet` by being emitted first.

    The trap was already in the tree: `parse_review` skipped an index it had read, so the
    surviving verdict WAS the emission order. A merge written as "keep the first" would
    reproduce that silently and pass any single-order test, which is why both orders are
    asserted here over the same two answers — `met` then `unmet`, and `unmet` then `met` —
    and both must say `unmet`.

    All-`met` is the other half: a clause that IS satisfied over both subjects has to come
    out `met`, or the fix trades a false refusal for a false block.
    """
    def merged_verdict(rows):
        chunks = [[1, 2, 3], [4, 5, 6]]
        # `merge_grading_chunks` takes the grader's answer OBJECTS, not the
        # `run_grader` envelope, which is what `grade_vault` hands it.
        merged, why = RV.merge_grading_chunks(
            chunks, [_met_answer([1, 2, 3])["structured"], _answer(rows)["structured"]])
        assert why == "", why
        five = [c for c in merged["clauses"] if c["clause"] == 5]
        assert len(five) == 1, merged["clauses"]
        return five[0]["verdict"], five[0]["sub_verdicts"]

    met_first = merged_verdict([(4, "met", "a"), (5, "met", "management"),
                                (5, "unmet", "consolidation"), (6, "met", "c")])
    unmet_first = merged_verdict([(4, "met", "a"), (5, "unmet", "consolidation"),
                                  (5, "met", "management"), (6, "met", "c")])
    assert met_first[0] == "unmet", met_first
    assert unmet_first[0] == "unmet", unmet_first
    assert [v["verdict"] for v in met_first[1]] == [v["verdict"] for v in unmet_first[1]][::-1], \
        "both orders must survive, in the order the grader emitted them"

    both_met, why = RV.merge_grading_chunks(
        [[1, 2, 3], [4, 5, 6]],
        [_met_answer([1, 2, 3])["structured"],
         _answer([(4, "met", "a"), (5, "met", "management"), (5, "met", "consolidation"),
                  (6, "met", "c")])["structured"]])
    assert why == "" and both_met is not None
    assert [c["verdict"] for c in both_met["clauses"]] == ["met"] * 6, both_met["clauses"]


def test_the_missing_clause_arm_still_refuses_after_the_duplicate_arm_opened(
        isolated, monkeypatch):
    """#2442 clause 5: weakening the duplicate arm must not weaken the gap arm.

    Both faults came out of one `if missing or repeated`, so the edit that stopped
    refusing a repeat could have stopped refusing a gap without anyone noticing — and the
    gap is the dangerous one: a five-row answer to a six-clause contract would become a
    verdict, and `vault_round` would land an item whose last clause nobody read. Clause 5
    of #2434's own review called that clause conjunctive and half-met, which is only a
    useful sentence if the OTHER clauses were graded at all.

    So the same shrink path that swallows a repeat still returns `GRADER_INCOMPLETE`,
    still names the missing index, and does it at the fixed point (1 clause per call).
    """
    _six_clause_item(isolated, 605, tmp_path=isolated)
    def gap(**kw):
        slice_ = _asked_clauses(kw["prompt"])
        return _answer([(i, "met", "graded") for i in slice_ if i != 6]
                       + [(5, "met", "answered twice on purpose"),
                          (5, "met", "and the second is fine")])
    monkeypatch.setattr(RV, "run_grader", gap)
    kind, why, clauses = RV.grade_vault(item_id=605, paths=["skills/x/SKILL.md"],
                                        diff="+x", vault=isolated)
    assert kind == RV.GRADER_INCOMPLETE and clauses == [], (kind, why, clauses)
    assert "no verdict for clause(s) 6" in why, why
    assert "two verdicts" not in why, why



def test_a_grader_that_answers_every_call_at_once_still_merges(
        isolated, monkeypatch, tmp_path):
    """Redundant answers are dropped, not counted as duplicates: a grader that
    grades all six in both calls still lands a complete contract. Refusing it
    would refuse the round for the grader being over-eager."""
    _six_clause_item(isolated, 605, tmp_path)
    monkeypatch.setattr(RV, "run_grader", lambda **kw: _met_answer([1, 2, 3, 4, 5, 6]))
    kind, why, clauses = RV.grade_vault(item_id=605, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind == "pass", why
    assert [c["clause"] for c in clauses] == [1, 2, 3, 4, 5, 6]


def test_merge_grading_chunks_is_the_single_reader_of_both_sides():
    """The merge is a pure function so the bound and the completeness check are
    one measurement, not a prompt-side claim and a reader-side hope."""
    ok, why = RV.merge_grading_chunks([[1, 2], [3]],
                                      [{"clauses": [{"clause": 1}, {"clause": 2}]},
                                       {"clauses": [{"clause": 3}]}])
    assert why == "" and [c["clause"] for c in ok["clauses"]] == [1, 2, 3]
    # An unsound premise in any call is unsound overall, and a refused amendment
    # in any call is refused: one call's finding is a finding.
    merged, _ = RV.merge_grading_chunks(
        [[1], [2]], [{"premise": "sound", "clauses": [{"clause": 1}], "amendments_ok": True},
                     {"premise": "unsound", "clauses": [{"clause": 2}], "amendments_ok": False,
                      "amendments_note": "clause 2 asks for an amendment"}])
    assert merged["premise"] == "unsound" and merged["amendments_ok"] is False
    assert merged["amendments_note"] == "clause 2 asks for an amendment"
    bad, why = RV.merge_grading_chunks([[1], [2]], [{"clauses": []}, {"clauses": [{"clause": 2}]}])
    assert bad is None and "no verdict for clause(s) 1" in why, why


def test_a_grader_cut_off_mid_generation_answers_diverged_not_skipped(
        isolated, monkeypatch, tmp_path):
    """The `#2260` error verbatim, out of the ledger row at 2026-10-06T01:31:39Z.
    `skipped` is what `land()` used to receive, and it is the word that let the
    land commit with `review: skipped`; `diverged` is a named mechanism with a
    measured denominator, so `land` can make it blocking.

    The assertions about the text changed with #2341: this stub diverges at EVERY
    size, so the reported failure is now the one that survived shrinking — clause 1
    alone at one clause per call, after the 3-clause chunk — and it names the size
    it reached rather than which call of a fixed plan it was. A divergence the
    shrink could not get past is still `diverged` with no clauses."""
    _six_clause_item(isolated, 606, tmp_path)
    err = ('finalizer failed: generation diverged at 8192 tokens — every field this schema '
           'admits is capped, so the object running past them is malformed output, not a '
           'budget (\'{"premise":"sound","clauses":[{"clause":1,"verdict":"met"')
    asked = []
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: (asked.append(_asked_clauses(kw["prompt"])),
                                      {"ok": False, "error": err})[1])
    kind, why, clauses = RV.grade_vault(item_id=606, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind == RV.GRADER_DIVERGED and clauses == []
    assert asked == [[1, 2, 3], [1]], f"one re-ask, at one clause: {asked}"
    assert "clause(s) 1, 2, 3 asked at 3 clauses per call shrank to 1 clause per call" in why, why
    assert "answering clause(s) 1 (1 clause per call)" in why, why
    assert "[2 grading generations issued]" in why, why
    assert RV.generation_diverged(err) is True
    # The abstentions that are NOT this item's: a backend that 503s says nothing
    # about the contract's size, and stays a `skipped` the land can still make.
    assert RV.generation_diverged("engine unavailable: 503") is False
    monkeypatch.setattr(RV, "run_grader",
                        lambda **kw: {"ok": False, "error": "engine unavailable: 503"})
    assert RV.grade_vault(item_id=606, paths=["skills/x/SKILL.md"], diff="+x",
                          vault=isolated)[0] == "skipped"


# ── #2341: a slice the generation cannot hold is re-asked smaller, not reported ──
#
# Item #2335 (two paragraphs added to `skills/youtube-digest/SKILL.md`, five
# acceptance clauses) was refused twice in a row by `automod_vault_land` on
# 2026-10-07 with the grader producing no verdict at all, and the second refusal
# reverted the edits. Both of its ledger rows name the SAME chunk:
# `attempt: 1 kind: incomplete` at 07:34:24Z and `attempt: 2 kind: diverged` at
# 07:35:58Z, each "call 1 of 2" on "clause(s) 1, 2, 3", because chunking was
# attempt-independent and `VAULT_REVIEW_MAX = 2` counts a grader failure as one of
# the item's two attempts. What follows pins the re-ask that has to happen inside
# ONE `grade_vault` call so a grading the grader could not finish is never the
# thing that spends an attempt.

FIVE = [f"clause {i} of a five-clause vault contract" for i in range(1, 6)]


def _five_clause_item(isolated, item_id, tmp_path, clauses=None):
    """A #2335-shaped contract: five clauses, so the plan is [1,2,3] then [4,5]."""
    texts = list(clauses or FIVE)
    write_item(isolated, item_id, clauses=texts)
    _confirm(item_id, acceptance="; ".join(texts), clauses=texts, surface="vault")
    (isolated / "skills" / "x").mkdir(parents=True, exist_ok=True)
    (isolated / "skills" / "x" / "SKILL.md").write_text("x\n", encoding="utf-8")
    return texts


def _asked_from(texts):
    """A detector for which clauses ONE prompt names, for an arbitrary contract."""
    def detect(prompt):
        return [i for i, t in enumerate(texts, 1) if f"{i}. {t}" in prompt]
    return detect


def test_a_chunk_that_diverges_is_re_asked_one_clause_per_call_and_graded(
        isolated, monkeypatch, tmp_path):
    """#2341 clause 1 on #2335's own shape: a 3-clause chunk that cannot finish at
    8192 tokens is re-asked as single clauses INSTEAD of being reported, and the
    contract comes back complete — an ordinary review, five verdicts, no failure
    kind for a grader that merely needed a smaller question."""
    texts = _five_clause_item(isolated, 607, tmp_path)
    asked = []
    detect = _asked_from(texts)

    def diverges_on_three_clauses_only(**kw):
        slice_ = detect(kw["prompt"])
        asked.append(slice_)
        if len(slice_) == 3:
            return {"ok": False, "error": "finalizer failed: generation diverged at 8192 tokens"}
        return _met_answer(slice_)
    monkeypatch.setattr(RV, "run_grader", diverges_on_three_clauses_only)
    kind, why, clauses = RV.grade_vault(item_id=607, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind == "pass", why
    assert [c["clause"] for c in clauses] == [1, 2, 3, 4, 5], clauses
    assert all(c["verdict"] == "met" for c in clauses), clauses
    # The 3-clause slice diverged and came back as three single-clause asks; the
    # 2-clause slice never diverged, so it is asked once, at its planned size.
    assert asked == [[1, 2, 3], [1], [2], [3], [4, 5]], asked


def test_a_partially_answered_chunk_keeps_its_verdicts_and_asks_only_the_rest(
        isolated, monkeypatch, tmp_path):
    """#2341's own finding: `merge_grading_chunks` already KNEW which indices were
    skipped (`missing`), yet `grade_vault` threw the whole call away — #2335's first
    attempt graded clause 1 of "clause(s) 1, 2, 3" and lost it. The re-ask is of the
    indices with no verdict, so a graded clause is never billed twice."""
    texts = _five_clause_item(isolated, 608, tmp_path)
    asked = []
    detect = _asked_from(texts)

    def drops_the_tail_of_a_three_clause_answer(**kw):
        slice_ = detect(kw["prompt"])
        asked.append(slice_)
        return _met_answer(slice_[:1] if len(slice_) == 3 else slice_)
    monkeypatch.setattr(RV, "run_grader", drops_the_tail_of_a_three_clause_answer)
    kind, why, clauses = RV.grade_vault(item_id=608, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind == "pass", why
    assert [c["clause"] for c in clauses] == [1, 2, 3, 4, 5], clauses
    # Clause 1 is asked ONCE and never appears in the re-ask list: only 2 and 3 had
    # no verdict, and the 2-clause slice came back complete at its planned size.
    assert asked == [[1, 2, 3], [2], [3], [4, 5]], asked


def test_shrinking_terminates_bounded_by_the_contract_size(isolated, monkeypatch, tmp_path):
    """#2341 clause 4's bound, measured rather than asserted in prose: the re-ask
    tree splits a failing slice into strictly smaller disjoint ones, so the
    generations one `grade_vault` issues are at most `2 * n - 1` for an n-clause
    contract, and shrinking stops dead at one clause per call. The stub is the
    worst case the tree admits — every multi-clause ask diverges, and the LAST
    clause of the contract also diverges alone, so every sibling is asked first."""
    for n in range(1, 13):
        item_id = 620 + n
        texts = [f"clause {i} of a {n}-clause vault contract" for i in range(1, n + 1)]
        _five_clause_item(isolated, item_id, tmp_path, clauses=texts)
        asked = []
        detect = _asked_from(texts)

        def worst_case(last=n, **kw):
            slice_ = detect(kw["prompt"])
            asked.append(slice_)
            if len(slice_) > 1 or slice_ == [last]:
                return {"ok": False,
                        "error": "finalizer failed: generation diverged at 8192 tokens"}
            return _met_answer(slice_)
        monkeypatch.setattr(RV, "run_grader", worst_case)
        kind, why, clauses = RV.grade_vault(item_id=item_id, paths=["skills/x/SKILL.md"],
                                            diff="+x", vault=isolated)
        assert kind in RV.GRADER_FAILURE_KINDS, f"{n}: {kind} — {why}"
        assert clauses == []
        assert len(asked) <= 2 * n - 1, f"{n}-clause contract issued {len(asked)}: {asked}"
        assert all(len(slice_) >= 1 for slice_ in asked)


def test_a_single_clause_chunk_that_diverges_is_the_end_of_shrinking(
        isolated, monkeypatch, tmp_path):
    """One clause per call is the last size there is, so a one-clause contract gets
    exactly one generation and its divergence is reported, not retried forever. A
    3-clause contract is the same story one level up: the plan is already one call,
    the shrink splits it into three, and the second of those to fail ends it."""
    texts = _five_clause_item(isolated, 609, tmp_path, clauses=["the only clause"])
    asked = []
    detect = _asked_from(texts)

    def always_diverges(**kw):
        asked.append(detect(kw["prompt"]))
        return {"ok": False, "error": "finalizer failed: generation diverged at 8192 tokens"}
    monkeypatch.setattr(RV, "run_grader", always_diverges)
    kind, why, clauses = RV.grade_vault(item_id=609, paths=["skills/x/SKILL.md"], diff="+x",
                                       vault=isolated)
    assert kind == RV.GRADER_DIVERGED and clauses == []
    assert asked == [[1]], f"one clause per call is the fixed point: {asked}"
    assert "1 clause per call" in why, why
    # A clause the plan already sizes at one is not re-asked: three clauses are one
    # planned call, and after the split the first failing single ends the grading.
    asked.clear()
    _five_clause_item(isolated, 610, tmp_path, clauses=["one", "two", "three"])
    detect3 = _asked_from(["one", "two", "three"])
    monkeypatch.setattr(RV, "run_grader", lambda **kw: (asked.append(detect3(kw["prompt"])),
                                                        {"ok": False, "error":
                                                         "finalizer failed: generation "
                                                         "diverged at 8192 tokens"})[1])
    kind, why, _ = RV.grade_vault(item_id=610, paths=["skills/x/SKILL.md"], diff="+x",
                                  vault=isolated)
    assert kind == RV.GRADER_DIVERGED
    assert asked == [[1, 2, 3], [1]], asked
    assert "[2 grading generations issued]" in why, why


# ── #2459: the finalizer's token counts reach the caller that writes the row ──
#
# Three `vault_review` rows on 2026-10-08/09 (items 2411 twice at 14:05:54Z and
# 15:03:15Z, 2453 at 2026-10-09T03:24:16Z) each say `generation diverged at 8192
# tokens` and none of them can say whether the 8192 went to reasoning or to
# malformed output. The numbers were already there: `app/routers/messages.py`
# puts `finalizer_output_tokens` and `finalizer_reasoning_tokens` on the `done`
# frame, and `_grade_once`'s `done` handler read four keys and dropped them.
# These nodes pin the read, and pin that what reaches the caller is the counts of
# the generation that FAILED rather than a sum over #2341's re-asks.

def test_the_done_frames_finalizer_token_counts_reach_the_graders_report(
        tmp_path, monkeypatch):
    """#2459 clause 1: the read in `_grade_once`, over the frame as the backend sends it.

    A diverged grading turn, which is the shape the ledger row is written for: the
    finalizer errored, so `structured` is None and `structured_error` carries the
    `generation diverged at 8192 tokens` text — and `app/harness/finalizer.py`'s
    diverged branch returns its usage dict alongside the error, so the counts ride
    out on the same frame. Only `_post_stream` is replaced, so the payload build,
    the event loop and the `done` handler all run.
    """
    frames = [("done", {
        "response": "", "stop_reason": "stop", "structured": None,
        "structured_error": "finalizer failed: generation diverged at 8192 tokens",
        "finalizer_output_tokens": 8192, "finalizer_reasoning_tokens": 6117})]
    monkeypatch.setattr(RV, "_post_stream", lambda *a, **k: list(frames))
    rep = RV.run_grader(prompt="grade this", item_id=2411, round_id="SM_T",
                        backend="http://127.0.0.1:9", sessions_dir=tmp_path,
                        model="m", max_turns=1, timeout=5)
    assert rep["ok"] is False and rep["structured"] is None, rep
    assert "generation diverged at 8192 tokens" in rep["structured_error"], rep
    assert rep["finalizer_output_tokens"] == 8192, rep
    assert rep["finalizer_reasoning_tokens"] == 6117, rep


def test_a_grading_turn_with_no_finalizer_usage_reports_nothing_and_raises_nothing(
        tmp_path, monkeypatch):
    """The other side of the same read: a `done` frame that carries neither key.

    A turn that never reached the finalizer — `finalizer_output_tokens` is emitted
    only when `options.final_schema` is set (`app/routers/messages.py`), and a
    timeout or transport error means there is no `done` frame at all. A `.get` that
    raised, or that wrote `None` into the ledger where a count belongs, is the
    difference between a row that says "no measurement" and a row that lies about
    one, which is the question #2459 exists to answer.
    """
    monkeypatch.setattr(RV, "_post_stream", lambda *a, **k: [
        ("done", {"response": "hi", "stop_reason": "stop",
                  "structured": {"clauses": []}, "structured_error": ""})])
    rep = RV.run_grader(prompt="grade this", item_id=2411, round_id="SM_T",
                        backend="http://127.0.0.1:9", sessions_dir=tmp_path,
                        model="m", max_turns=1, timeout=5)
    assert "finalizer_output_tokens" not in rep, rep
    assert "finalizer_reasoning_tokens" not in rep, rep


def _diverging_grader(counts):
    """A `run_grader` stub: every generation diverges, and generation *n* spent
    `counts[n]` = `(output_tokens, reasoning_tokens)`. `counts` is consumed in call
    order, and the last entry repeats — the shrink asks more generations than there
    are entries whenever the contract is bigger than the list.

    The marker goes in `error`, the key `ask` classifies on (review.py:1983), and in
    `structured_error`, the key `_grade_once` fills when the finalizer is the thing
    that failed — on a real frame the two carry the same text.
    """
    state = {"i": 0}

    def grader(**kw):
        i = min(state["i"], len(counts) - 1)
        state["i"] += 1
        out, reason = counts[i]
        why = f"finalizer failed: generation diverged at {out} tokens"
        return {"ok": False, "error": why, "structured": None, "structured_error": why,
                "finalizer_output_tokens": out, "finalizer_reasoning_tokens": reason}
    return grader


def test_a_diverged_grading_hands_the_caller_the_failing_generations_counts(
        isolated, monkeypatch, tmp_path):
    """#2459 clause 3, at the seam that decides it: `grade_vault` over a shrink.

    Three clauses are one planned call; it diverges at 8192, #2341 re-asks clause 1
    alone and that diverges too at 311 tokens. The row has to answer "was one
    generation's budget the problem", so it carries 311 — the generation whose
    error text is in `why` — and not 8503, which is what summing the two would say
    about a budget the model was never given.
    """
    _five_clause_item(isolated, 610, tmp_path, clauses=["one", "two", "three"])
    monkeypatch.setattr(RV, "run_grader",
                        _diverging_grader([(8192, 6117), (311, 44)]))
    usage: dict = {}
    kind, why, clauses = RV.grade_vault(item_id=610, paths=["skills/x/SKILL.md"],
                                        diff="+x", vault=isolated, usage_out=usage)
    assert kind == RV.GRADER_DIVERGED and clauses == [], (kind, why)
    assert usage == {"output_tokens": 311, "reasoning_tokens": 44}, usage
    assert usage["output_tokens"] != 8192 + 311, "a sum over re-asks is not a budget"


def test_a_grading_that_answers_every_clause_leaves_the_usage_out_param_empty(
        isolated, monkeypatch, tmp_path):
    """A success row gets no counts, so it gets no keys.

    #2459 records usage on a row that reports a grader FAILURE. The success path
    stays exactly the row it is today — the out-param is untouched — because a
    completed grading's token spend is not the measurement the ledger is missing,
    and writing it on every landing would change 392 existing rows' shape for a
    question nobody is asking.
    """
    _five_clause_item(isolated, 610, tmp_path, clauses=["one", "two", "three"])
    # Every clause in every answer: the shrink may ask any subset, and a `pass` is
    # only reachable when nothing is left ungraded.
    monkeypatch.setattr(RV, "run_grader", lambda **kw: _met_answer([1, 2, 3]))
    usage = {"untouched": True}
    kind, why, clauses = RV.grade_vault(item_id=610, paths=["skills/x/SKILL.md"],
                                        diff="+x", vault=isolated, usage_out=usage)
    assert kind == "pass", why
    assert usage == {"untouched": True}, "a pass writes no usage onto the out-param"


def test_grade_vault_without_an_out_param_still_returns_the_verdicts(
        isolated, monkeypatch, tmp_path):
    """The out-param is optional, and 64 existing callers rely on that.

    Every grade_vault call in this file, in `tests/test_automod_gate.py`,
    `test_automod_vault_round.py`, `test_review_grader_determinism.py` and
    `test_automod_review_cli_rendering.py` unpacks three values, and
    `agent_mcp/automod.py:919` hands `review.grade_vault` to
    `vault_round.GRADER` directly. This is the shape check for that: a call with no
    `usage_out` returns the same 3-tuple it always did, on the failing path too.
    """
    _five_clause_item(isolated, 610, tmp_path, clauses=["one", "two", "three"])
    monkeypatch.setattr(RV, "run_grader", _diverging_grader([(8192, 6117)]))
    out = RV.grade_vault(item_id=610, paths=["skills/x/SKILL.md"], diff="+x",
                         vault=isolated)
    assert len(out) == 3, out
    assert out[0] == RV.GRADER_DIVERGED and out[2] == [], out


def test_recording_the_counts_changes_no_budget_and_adds_no_thinking_knob():
    """#2459 clause 5: this change is a read, not a knob.

    The whole point of recording the counts is to find out whether 8192 is being
    spent on reasoning or on malformed output, and you cannot learn that from a run
    whose budget or thinking setting you also changed. Two halves, both measured from
    the tree: the finalizer's budget is still the 8192 `config.yaml` states, and
    neither module that now CARRIES the counts mentions a thinking knob in any of its
    spellings — `app/harness/tests/test_finalizer.py`'s
    `test_the_finalizer_sends_no_thinking_knob_in_either_spelling` still pins the
    payload itself, and stays green unchanged.
    """
    import yaml
    repo_root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["harness"]["finalizer"]["max_tokens"] == 8192, "budget unchanged"
    for rel in ("scripts/automod/review.py", "scripts/automod/vault_round.py"):
        src = (repo_root / rel).read_text(encoding="utf-8")
        for key in ("reasoning_effort", "chat_template_kwargs", "enable_thinking",
                    "thinking_token_budget"):
            assert key not in src, f"{rel} mentions {key}"
