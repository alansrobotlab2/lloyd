"""The review rung refused 86% of what it graded. This is why, and the fix.

Over 2026-09-11 the rung graded 14 autocode rounds and blocked six of them
twice each, ending the round. The findings were real; most were also
unfixable *inside* the round:

  * a clause that can only be observed with live traffic, which had no word
    of its own and came out `partial` — a refusal for a round that did the
    work (#859, refused twice with the mechanism complete on both commits);
  * a seam across an HTTP boundary that no test in this repo can cross until
    the change is live;
  * "test files changed but no test function was added", on rounds that
    tightened existing assertions;
  * five `met`s downgraded for an `evidence_path` that was real but lived in
    the vault;
  * an amendment answered from the ledger and never re-graded (866-c);
  * a grader 503 caused by a *sibling* round's landing (866-a).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.automod import review as RV


def _parsed(**kw):
    base = dict(premise="sound", clauses=[], test_honesty=[],
                seams_unverified=[], downgraded=[], summary="ok")
    base.update(kw)
    return base


def _clause(idx, verdict, **kw):
    c = dict(clause=idx, verdict=verdict, evidence_path="app/x.py",
             evidence_line=1, test_node_id="tests/t.py::test_x",
             how_verified="ran", note="n")
    c.update(kw)
    return c


# ---------------------------------------------------------------------------
# post_landing
# ---------------------------------------------------------------------------

def test_post_landing_is_a_clause_verdict():
    assert "post_landing" in RV.CLAUSE_VERDICTS
    enum = RV.REVIEW_SCHEMA["properties"]["clauses"]["items"]["properties"]["verdict"]["enum"]
    assert enum == list(RV.CLAUSE_VERDICTS), "schema is built from the one list"


def test_a_post_landing_clause_does_not_refuse_the_round():
    """#859's exact shape: the mechanism is in the diff and correct, and the
    clause needs traffic that does not exist until it is live.
    """
    kind, findings = RV.decide(
        _parsed(clauses=[_clause(1, "post_landing", note="needs a day of traffic")]), [])
    assert kind == "pass"
    assert "after landing" in findings
    assert "needs-human" in findings


def test_a_post_landing_clause_needs_a_pinned_mechanism(tmp_path):
    """Without evidence it is a claim about a mechanism nobody has seen —
    which is exactly "not done", and must not land as though it were.
    """
    obj = {"premise": "sound", "clauses": [
        {"clause": 1, "verdict": "post_landing", "evidence_path": "",
         "evidence_line": 0, "test_node_id": "", "how_verified": "inferred",
         "note": "later"}]}
    parsed = RV.parse_review(obj, worktree=tmp_path, changed_tests=[], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "partial"
    assert any("post_landing without a pinned mechanism" in d
               for d in parsed["clauses"][0]["downgraded"])


def test_a_post_landing_clause_with_a_real_path_survives(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x = 1")
    obj = {"premise": "sound", "clauses": [
        {"clause": 1, "verdict": "post_landing", "evidence_path": "app/x.py",
         "evidence_line": 1, "test_node_id": "", "how_verified": "read",
         "note": "needs traffic"}]}
    parsed = RV.parse_review(obj, worktree=tmp_path, changed_tests=[], n_clauses=1)
    assert parsed["clauses"][0]["verdict"] == "post_landing"


def test_unmet_still_refuses_beside_a_post_landing():
    kind, _ = RV.decide(_parsed(clauses=[
        _clause(1, "post_landing"), _clause(2, "unmet")]), [])
    assert kind == "retry"


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------

def test_a_seam_blocks_on_the_first_attempt():
    """Blocking once is what keeps it honest: the author is told, with a
    chance to add the test.
    """
    kind, findings = RV.decide(
        _parsed(seams_unverified=[{"seam": "loopback POST",
                                   "testable_before_landing": True}]),
        [], attempt=1)
    assert kind == "retry"
    assert "seam unverified" in findings


def test_a_seam_advises_on_the_second():
    kind, findings = RV.decide(
        _parsed(seams_unverified=[{"seam": "loopback POST",
                                   "testable_before_landing": True}]),
        [], attempt=2)
    assert kind == "pass"
    assert "not refusing again" in findings
    assert "loopback POST" in findings, "the finding still rides into the report"


@pytest.mark.parametrize("policy,attempt,blocks", [
    ("first", 1, True), ("first", 2, False), ("first", 3, False),
    ("always", 1, True), ("always", 2, True),
    ("never", 1, False), ("never", 2, False),
    ("", 1, True),          # empty falls back to `first`
    ("nonsense", 1, True),  # so does an unknown policy
])
def test_the_seam_policy_table(policy, attempt, blocks):
    assert RV.seams_block(policy, attempt) is blocks


def test_an_untestable_seam_never_blocks_even_on_attempt_one():
    kind, findings = RV.decide(
        _parsed(seams_unverified=[{"seam": "a real pool tick",
                                   "testable_before_landing": False}]),
        [], attempt=1)
    assert kind == "pass"
    assert "post-landing seam" in findings


# ---------------------------------------------------------------------------
# precheck severities
# ---------------------------------------------------------------------------

def test_the_dishonest_test_patterns_still_block():
    for pat, why, sev in RV._HONESTY_PATTERNS:
        assert sev == "blocking", why
    kind, _ = RV.decide(_parsed(), [
        {"file": "tests/t.py", "line": 3, "problem": "`or True` …",
         "severity": "blocking"}])
    assert kind == "retry"


def test_no_new_test_function_is_advisory():
    """A round that tightens an existing test's assertions has pinned exactly
    what it should and added no `def test_`.
    """
    assert RV._NO_NEW_TEST_SEVERITY == "advisory"
    kind, findings = RV.decide(_parsed(), [
        {"file": "tests/t.py", "line": 0,
         "problem": "test files changed but no test function was added while "
                    "the item has acceptance clauses to pin",
         "severity": "advisory"}])
    assert kind == "pass"
    assert "advisory" in findings


def test_a_precheck_with_no_severity_still_blocks():
    """An entry written before severities existed keeps the old reading."""
    kind, _ = RV.decide(_parsed(), [
        {"file": "tests/t.py", "line": 3, "problem": "something"}])
    assert kind == "retry"


def test_a_dishonest_pattern_precheck_is_blocking(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text(
        "def test_a():\n    assert 1 == 1 or True\n")
    monkeypatch.setattr(RV, "_git", lambda *a, **k: "")
    out = RV.honesty_prechecks(tmp_path, "BASE", ["tests/t.py", "app/x.py"],
                               n_clauses=1)
    or_true = next(o for o in out if "or True" in o["problem"])
    assert or_true["severity"] == "blocking"


def test_the_no_new_test_precheck_is_advisory(tmp_path, monkeypatch):
    """The round that tightens an existing test's assertions: a real
    observation and a bad refusal.
    """
    (tmp_path / "tests").mkdir()
    # Post has the same test function as pre, with a tighter assertion — so
    # `added_tests` is 0 and this is the shape that used to refuse.
    (tmp_path / "tests" / "t.py").write_text(
        "def test_a():\n    assert compute() == 7\n")
    monkeypatch.setattr(
        RV, "_git",
        lambda *a, **k: "def test_a():\n    assert compute() is not None\n")
    out = RV.honesty_prechecks(tmp_path, "BASE", ["tests/t.py", "app/x.py"],
                               n_clauses=1)
    no_test = next(o for o in out if "no test function was added" in o["problem"])
    assert no_test["severity"] == "advisory"
    # ...and it does not refuse.
    kind, _ = RV.decide(_parsed(), out)
    assert kind == "pass"


# ---------------------------------------------------------------------------
# evidence roots
# ---------------------------------------------------------------------------

def test_a_worktree_path_still_wins(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text("x")
    assert RV.normalize_evidence_path("app/x.py", tmp_path) == "app/x.py"


def test_a_vault_relative_path_resolves_under_the_roots(tmp_path):
    root = tmp_path / "vault"
    (root / "lloyd").mkdir(parents=True)
    (root / "lloyd" / "SOUL.md").write_text("soul")
    got = RV.normalize_evidence_path("lloyd/SOUL.md", tmp_path / "wt",
                                     roots=(root,))
    assert got == str(root / "lloyd" / "SOUL.md")


def test_a_line_suffix_is_still_stripped_under_the_roots(tmp_path):
    root = tmp_path / "vault"
    (root / "lloyd").mkdir(parents=True)
    (root / "lloyd" / "SOUL.md").write_text("soul")
    got = RV.normalize_evidence_path("lloyd/SOUL.md:44-51", tmp_path / "wt",
                                     roots=(root,))
    assert got.endswith("SOUL.md")


def test_a_path_in_neither_place_is_still_empty(tmp_path):
    assert RV.normalize_evidence_path("nowhere/at/all.py", tmp_path,
                                      roots=(tmp_path / "vault",)) == ""


def test_the_default_roots_name_the_vault():
    assert any(str(r).endswith("obsidian") for r in RV.REVIEW_EVIDENCE_ROOTS)


def _prompt_text() -> str:
    return RV.build_prompt(
        contract={"clauses": ["c1"], "amendments": [], "human_clauses": [],
                  "id": 1, "title": "t", "body": "b"},
        diff="diff", diff_truncated=False, changed_tests=["tests/t.py"],
        test_counts={"passed": 1}, worktree=Path("/wt"),
        run_tests=Path("/wt/run"))


def test_the_prompt_no_longer_forbids_what_the_parser_accepts():
    """The prompt said paths are worktree-relative "not `~/…`" while the
    parser already accepted them, and five `met`s were downgraded on
    2026-09-11 for paths that were real.
    """
    text = _prompt_text()
    assert "not `~/…`" not in text
    assert "a vault path you actually read is" in text


def test_the_prompt_teaches_post_landing_rather_than_unsatisfiable():
    """The two are opposite remedies: `post_landing` lands the change and
    waits for a person; `unsatisfiable` refuses and amends the contract.
    Telling the grader to use the second for the first is what parked #859.
    """
    text = _prompt_text()
    assert "is `post_landing`" in text
    assert "does not refuse the round" in text
    # `unsatisfiable` is still taught, for the contract-defect case.
    assert "no diff could EVER satisfy" in text


# ---------------------------------------------------------------------------
# the grader is not always up
# ---------------------------------------------------------------------------

def _stream_503(*_a, **_k):
    import urllib.error
    raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)


def test_a_503_before_the_stream_opens_is_retried(tmp_path, monkeypatch):
    """Round 866-a: the grader hit a 503 because ANOTHER round was landing at
    that moment. The rung recorded `external`, the turn ended without ever
    re-gating, and it was reaped 30 minutes later.
    """
    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            _stream_503()
        yield "done", {"response": "r", "stop_reason": "stop",
                       "structured": {"premise": "sound", "clauses": []}}

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")

    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert rep["ok"] is True
    assert rep["retries"] == 2
    assert attempts["n"] == 3


def test_the_session_id_is_reused_across_retries(tmp_path, monkeypatch):
    """Three retries must read as one grading of one commit, not as three."""
    minted = []

    def _mint(*a, **k):
        minted.append(1)
        return f"sess-{len(minted)}"

    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            _stream_503()
        yield "done", {"response": "r", "stop_reason": "stop",
                       "structured": {"premise": "sound", "clauses": []}}

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", _mint)
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert len(minted) == 1
    assert rep["session_id"] == "sess-1"


def test_a_failure_mid_stream_is_never_retried(tmp_path, monkeypatch):
    """Once events have arrived the turn ran and cost the round. Re-POSTing
    would run a second grading turn whose verdict duplicates a judgment
    already partly made.
    """
    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        yield "token", {"text": "thinking"}
        _stream_503()

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert rep["ok"] is False
    assert attempts["n"] == 1
    assert rep["retries"] == 0


def test_a_non_availability_error_is_not_retried(tmp_path, monkeypatch):
    """A 400 is a refusal of THIS request and waiting changes nothing."""
    import urllib.error
    attempts = {"n": 0}

    def _stream(url, payload, timeout):
        attempts["n"] += 1
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
        yield  # pragma: no cover

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: None)
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=600)
    assert attempts["n"] == 1
    assert rep["retries"] == 0


def test_the_wait_is_bounded(tmp_path, monkeypatch):
    slept = []

    def _stream(url, payload, timeout):
        _stream_503()
        yield  # pragma: no cover

    monkeypatch.setattr(RV, "_post_stream", _stream)
    monkeypatch.setattr(RV.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(RV, "write_session", lambda *a, **k: "sess-1")
    rep = RV.run_grader(prompt="p", item_id=1, round_id="R", backend="http://x",
                        sessions_dir=tmp_path, unavailable_wait_s=60)
    assert rep["ok"] is False
    assert sum(slept) <= 60
    assert "still unavailable" in rep["error"]


@pytest.mark.parametrize("err,expected", [
    ("HTTP 503: lloyd is landing a code update; retry in 42s", 42.0),
    ("HTTP 503: retry in 9999s", 60.0),        # capped
    ("HTTP 503: no hint at all", 15.0),        # first backoff
])
def test_the_retry_delay_reads_the_servers_own_hint(err, expected):
    assert RV._retry_delay(err, 0) == expected


@pytest.mark.parametrize("err,unavailable", [
    ("HTTP 503: Service Unavailable", True),
    ("URLError: Connection refused", True),
    ("lloyd is landing a code update; not starting a worker turn", True),
    ("HTTP 400: Bad Request", False),
    ("HTTP 500: Internal Server Error", False),
    ("", False),
])
def test_what_counts_as_unavailable(err, unavailable):
    assert RV._is_unavailable(err) is unavailable


# ---------------------------------------------------------------------------
# human_paths, and the post_landing round trip onto the item
# ---------------------------------------------------------------------------

from scripts.automod import backlog as B  # noqa: E402
from scripts.automod import spec as SP    # noqa: E402


def test_a_scope_refusal_names_the_move():
    """A round told only that it may not have a path reached for `git add -f`,
    which defeats the check rather than reporting past it.
    """
    ok, why, _ = SP.check_scope([".gitignore"])
    assert ok is False
    assert "human_paths" in why
    assert "git add -f" in why
    assert "state dir" in why


def test_human_paths_survive_parse_outcome():
    out = B.parse_outcome({
        "acceptance": "met", "landed": True, "deferred_to": [], "summary": "s",
        "spawned": [], "clause_outcomes": [],
        "human_paths": [{"path": ".gitignore", "reason": "needs a rule"}]})
    assert out["human_paths"] == [{"path": ".gitignore", "reason": "needs a rule"}]


def test_the_outcome_schema_offers_human_paths():
    props = B.IMPLEMENT_OUTCOME_SCHEMA["properties"]
    assert "human_paths" in props
    assert props["human_paths"]["items"]["required"] == ["path", "reason"]
    assert "maxLength" not in json_dumps(B.IMPLEMENT_OUTCOME_SCHEMA)


def json_dumps(x):
    import json
    return json.dumps(x)


def test_apply_post_landing_rescues_a_deferred_clause():
    """The review rung decides a clause is observable only after landing; the
    implementer, inside the round, cannot know that and honestly says
    `deferred`. Re-read, the round has done its job.
    """
    out = {"acceptance": "not_met", "clause_outcomes": [
        {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []},
        {"clause": 2, "outcome": "deferred", "evidence": "", "deferred_to": []},
    ]}
    got = B.apply_post_landing(out, [2])
    assert got["acceptance"] == "met"
    assert got["post_landing_clauses"] == [2]
    assert got["clause_outcomes"][1]["outcome"] == "met"
    assert got["clause_outcomes"][1]["post_landing"] is True


def test_apply_post_landing_never_rescues_an_unrelated_clause():
    out = {"acceptance": "not_met", "clause_outcomes": [
        {"clause": 1, "outcome": "not_met", "evidence": "", "deferred_to": []},
        {"clause": 2, "outcome": "deferred", "evidence": "", "deferred_to": []},
    ]}
    got = B.apply_post_landing(out, [2])
    assert got["acceptance"] == "not_met", "clause 1 is still genuinely not met"


def test_apply_post_landing_leaves_a_met_claim_alone():
    """A clause the implementer called `met` stands on its own and needs no
    rescue — marking it post_landing would hold an item open for nothing.
    """
    out = {"acceptance": "met", "clause_outcomes": [
        {"clause": 1, "outcome": "met", "evidence": "t", "deferred_to": []}]}
    got = B.apply_post_landing(out, [1])
    assert "post_landing_clauses" not in got
    assert got is out


def test_apply_post_landing_is_a_noop_with_no_marks():
    out = {"acceptance": "deferred", "clause_outcomes": []}
    assert B.apply_post_landing(out, []) is out
    assert B.apply_post_landing(None, [1]) is None
