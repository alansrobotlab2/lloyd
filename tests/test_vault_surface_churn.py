"""A vault item whose fix is live must not keep coming back.

#575 (2026-09-14): a `vault` item whose fix landed at 21:25Z through
`automod_vault_land`, graded all five clauses `met` by the vault review and
then by three code reviews — and re-offered anyway. Four things kept it in the
loop, and each has a test here that fails without its fix:

  * the implement prompt told every surface to `automod_start` and pin each
    clause with a test, so the vault turn cut a code round for a test file
    and spent its 150 iterations gating it;
  * `settled_landings` read a finished row with a `round_id` only as a code
    landing, so the vault landing beside a never-promoted round was invisible;
  * a turn that died at `max_turns` had no outcome, and nothing read the vault
    review's own per-clause verdicts in its place;
  * the settle sweep ran only in housekeeping, so the turn-end reconcile put
    the item back in `up_next` and the next round took it two minutes later.

Plus the re-offer rule it exposed: "it comes back once" re-offered a second
budget death too.
"""
from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, review as RV, state as S, vault_round as V
from workers.sources import _common as C
from workers.sources import autocode as I


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    return d


def write_item(d: Path, item_id, *, status="up_next", clauses=None) -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created, "board": "lloyd",
          "tags": ["backlog"]}
    if clauses:
        fm["acceptance_clauses"] = list(clauses)
    path = d / f"{item_id}-a-thing.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# A thing\n\nDo it.\n",
                    encoding="utf-8")
    return path


def _ev(**kw):
    S.append_event(kw, path=S.LEDGER_PATH)


def _events(kind):
    return [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == kind]


def _fm(path):
    return B._split_frontmatter(path.read_text())[0]


MET5 = [{"clause": i, "verdict": "met"} for i in range(1, 6)]


def _vault_turn(item_id, *, round_id="SM_575", commits=("588339ab0000",), graded=MET5,
                surface="vault", outcome=None, stop_reason="max_turns"):
    """One implement turn as #575's first one was recorded: a vault landing
    whose review passed, a code round beside it, and no outcome."""
    _ev(event="backlog_implement", item_id=item_id, phase="started")
    for sha in commits:
        _ev(event="vault_land", ok=True, item_id=item_id, commit=sha, review="pass",
            review_clauses=list(graded) if graded is not None else None)
    fin = {"event": "backlog_implement", "item_id": item_id, "phase": "finished",
           "round_id": round_id, "vault_commits": list(commits), "surface": surface,
           "stop_reason": stop_reason, "num_turns": 151, "outcome": outcome}
    _ev(**fin)


# ── the landing beside a tests-only round is a landing ─────────────────────

def test_575s_shape_closes_on_its_vault_review(isolated):
    p = write_item(isolated, 575)
    _vault_turn(575)
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 575, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == "588339ab0000"
    assert "vault review graded every acceptance clause met" in fm["activity_log"][-1]
    ev = _events("item_landed")[-1]
    assert ev["vault"] is True and ev["acceptance_source"] == "vault_review"


def test_without_the_reviews_verdicts_the_same_turn_is_still_invisible(isolated):
    """The counterfactual, and the shape of every landing recorded before
    `review_clauses` existed: nothing to stand in for the missing outcome."""
    p = write_item(isolated, 575)
    _vault_turn(575, graded=None)
    assert B.close_settled_items(S.LEDGER_PATH) == []
    assert "automod_landed" not in _fm(p)


@pytest.mark.parametrize("verdict", ["partial", "not_met", "post_landing", "ungraded"])
def test_one_clause_short_of_met_is_not_a_landing(isolated, verdict):
    p = write_item(isolated, 576)
    graded = MET5[:4] + [{"clause": 5, "verdict": verdict}]
    _vault_turn(576, graded=graded)
    assert B.close_settled_items(S.LEDGER_PATH) == []
    assert _fm(p)["status"] == "up_next"


def test_a_gap_in_the_graded_clauses_is_not_met(isolated):
    write_item(isolated, 577)
    _vault_turn(577, graded=[{"clause": 1, "verdict": "met"}, {"clause": 3, "verdict": "met"}])
    assert B.vault_review_outcome(S.LEDGER_PATH, ["588339ab0000"]) is None
    assert B.close_settled_items(S.LEDGER_PATH) == []


@pytest.mark.parametrize("surface", ["code", "mixed", "frontend"])
def test_only_a_vault_items_round_can_land_on_the_vault_review(isolated, surface):
    """A mixed item's clauses are the code gate's; its vault half landing
    says nothing about the code it still owes."""
    write_item(isolated, 578)
    _vault_turn(578, surface=surface)
    assert B.close_settled_items(S.LEDGER_PATH) == []


def test_the_surface_falls_back_to_the_triage_verdict(isolated):
    write_item(isolated, 579)
    _ev(event="backlog_triage", item_id=579, verdict="confirmed", surface="vault",
        acceptance="a", acceptance_clauses=["a"])
    _vault_turn(579, surface="", graded=[{"clause": 1, "verdict": "met"}])
    assert B.close_settled_items(S.LEDGER_PATH)[0]["closed"] is True


def test_a_promoted_round_waits_for_its_settle_and_then_closes_on_the_review(isolated):
    """Promoted, not yet settled: the guardian is still watching the code, and
    the vault review does not jump that window. Once it settles, the code
    landing is the one recorded — and a `vault` item's missing outcome is
    still the vault review's to fill, or whether the item closes would depend
    on whether the promoter reached `promoted` before the sweep ran."""
    p = write_item(isolated, 580)
    _vault_turn(580, round_id="SM_580")
    _ev(event="promoted", round_id="SM_580", commit="c0de0000")
    assert B.close_settled_items(S.LEDGER_PATH) == []
    _ev(event="settled", commit="c0de0000")
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 580, "closed": True, "acceptance": "met"}]
    assert _fm(p)["automod_landed"] == "c0de0000", "the code landing, not the vault one"
    assert _events("item_landed")[-1]["acceptance_source"] == "vault_review"


def test_a_code_items_promotion_never_borrows_a_vault_review(isolated):
    write_item(isolated, 596)
    _vault_turn(596, round_id="SM_596", surface="mixed")
    _ev(event="promoted", round_id="SM_596", commit="c0de0596")
    _ev(event="settled", commit="c0de0596")
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 596, "closed": False, "acceptance": None}]


def test_the_no_round_path_checks_the_surface_too(isolated):
    """`grade_vault` grades an item with no recorded surface at all, so the
    surface has to be checked where the verdict is read, on every path."""
    write_item(isolated, 597)
    _vault_turn(597, round_id=None, surface="code")
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 597, "closed": False, "acceptance": None}]


#: A turn's own refusal that names what it refused: the half of #575 that keeps
#: standing. `clause_outcomes` carries a real `not_met`, so
#: `outcome_carries_no_claim` is False and the turn's word outranks the review.
REFUSED_CLAUSE = {"acceptance": "not_met", "landed": True, "deferred_to": [],
                  "summary": "clause 5 is not on disk", "spawned": [],
                  "clause_outcomes": [{"clause": 5, "outcome": "not_met",
                                       "evidence": "clause 5's file is absent"}]}

#: The same word with nothing behind it — #1896's finalizer (`finalizer_tokens:
#: 481`, `clause_outcomes: []`) 25 s after its vault review graded all five
#: clauses met, and #1725's at 2026-09-28T08:26Z. `outcome_carries_no_claim` is
#: True on this shape: the module's own predicate says it states nothing.
CLAIMLESS_NOT_MET = {"acceptance": "not_met", "landed": True, "deferred_to": [],
                     "summary": "", "spawned": []}


def test_an_outcome_the_turn_reported_is_never_overridden(isolated):
    """#575's guarantee in the form that means something: a refusal that names
    its clause outranks an all-`met` review, and the item is left open for it.

    Re-pointed by #1910. The fixture used to be `{"acceptance": "not_met", ...}`
    with no `clause_outcomes` key at all — byte-for-byte the shape
    `outcome_carries_no_claim` declares states nothing — so the node was pinning
    "a verdict naming no clause can never be overridden", which is the half of
    #575 #1910 exists to undo: that reading is what left #1896 open on a note
    naming no clause while its review had graded all five met. The half worth
    keeping is that a refusal the turn can substantiate is never overridden by
    the reviewer, and that is what this node now asserts.
    """
    p = write_item(isolated, 581)
    _vault_turn(581, round_id=None, stop_reason="stop", outcome=REFUSED_CLAUSE)
    assert B.vault_review_outcome(S.LEDGER_PATH, ["588339ab0000"]), \
        "the review graded all five met; the turn's named refusal still decides"
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 581, "closed": False, "acceptance": "not_met"}]
    fm = _fm(p)
    assert fm["status"] == "up_next", "the turn's word decides: left open, not closed"
    assert "clause(s) [5]" in fm["activity_log"][-1], \
        "the note names the clause it refused, not nothing"


def test_a_claim_less_not_met_beside_an_all_met_vault_review_closes_on_the_review(isolated):
    """#1910 clause 1: #1896's landing, 2026-09-30T12:03Z.

    The finalizer answered `not_met` with an empty `clause_outcomes` a quarter of
    a minute after the vault review had graded all five clauses met, and the sweep
    took the empty word: `item_landed` recorded `acceptance: not_met` with
    `acceptance_source: None`, the item moved to `draft` with its one unattended
    attempt spent, and the note read "reported the acceptance check not met"
    naming no clause. A hand re-check that morning found all five clauses true on
    disk. The review's verdict decides it now.
    """
    p = write_item(isolated, 1896)
    _vault_turn(1896, round_id=None, stop_reason="stop", outcome=CLAIMLESS_NOT_MET)
    assert B.outcome_carries_no_claim(CLAIMLESS_NOT_MET), \
        "the fixture is the shape this module itself calls a non-claim"
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 1896, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == "588339ab0000"
    ev = _events("item_landed")[-1]
    assert ev["acceptance"] == "met" and ev["acceptance_source"] == "vault_review", ev


def test_a_claim_less_not_met_beside_an_unpromoted_round_still_yields_a_row(isolated):
    """#1910 clause 3: the second vault path, #1725 (`afbb9251`, 2026-09-28T08:26Z).

    A vault turn that opened a code round which never promoted, reporting a
    claim-less `not_met`, produced *no* `item_landed` row at all: the guard on the
    second vault branch skipped it, so the landed vault commit was invisible to the
    sweep — no `automod_landed` marker, no activity note, nothing for a human to
    read. The all-met vault review decides it here too.
    """
    p = write_item(isolated, 1725)
    _vault_turn(1725, round_id="SM_1725", stop_reason="stop", outcome=CLAIMLESS_NOT_MET)
    assert _events("promoted") == [], "the tests-only round never promoted"
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 1725, "closed": True, "acceptance": "met"}], out
    ev = _events("item_landed")[-1]
    assert ev["commit"] == "588339ab0000", "the vault commit: the only thing that landed"
    assert ev["acceptance"] == "met" and ev["acceptance_source"] == "vault_review", ev
    assert _fm(p)["status"] == "done"


def test_a_refusal_that_names_its_clause_beside_an_unpromoted_round_yields_no_row(isolated):
    """The guard the clause-3 fix keeps standing: a refusal the turn can
    substantiate is still not the sweep's to override, so the vault landing
    beside an unpromoted round is still skipped and nothing is recorded."""
    p = write_item(isolated, 1726)
    _vault_turn(1726, round_id="SM_1726", stop_reason="stop", outcome=REFUSED_CLAUSE)
    assert B.close_settled_items(S.LEDGER_PATH) == []
    assert _events("item_landed") == []
    assert "automod_landed" not in _fm(p)


def test_a_round_and_a_reported_met_closes_only_with_the_reviews_agreement(isolated):
    """Before: a vault item whose tests round never promoted stayed open even
    when the turn itself said `met`. Now it closes — on the implementer's own
    outcome, and only because the vault review agreed."""
    met = {"acceptance": "met", "landed": False, "deferred_to": [], "summary": "skill fixed",
           "spawned": []}
    write_item(isolated, 582)
    _vault_turn(582, stop_reason="stop", outcome=met,
                graded=[{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "partial"}])
    assert B.close_settled_items(S.LEDGER_PATH) == []
    p = write_item(isolated, 583)
    _vault_turn(583, stop_reason="stop", outcome=met)
    assert B.close_settled_items(S.LEDGER_PATH)[0]["closed"] is True
    assert "the round reported the acceptance check met — skill fixed" in _fm(p)["activity_log"][-1]


def test_no_round_and_no_outcome_uses_the_review_too(isolated):
    """The turn never opened a round and still died at its budget."""
    p = write_item(isolated, 584)
    _vault_turn(584, round_id=None)
    assert B.close_settled_items(S.LEDGER_PATH)[0]["closed"] is True
    assert _fm(p)["status"] == "done"


def test_a_reverted_vault_commit_voids_the_review(isolated):
    write_item(isolated, 585)
    _vault_turn(585)
    _ev(event="vault_revert", reverted="588339ab0000", commit="beef0000", reason="manual")
    assert B.close_settled_items(S.LEDGER_PATH) == []


def test_only_the_newest_landing_decides(isolated):
    """An earlier all-met review describes a tree a later commit may have
    changed. The later commit's review skipped — a grader answering 503 during
    a landing drain, as #575's did at 22:48Z — grades nothing, so nothing
    closes."""
    partial = MET5[:4] + [{"clause": 5, "verdict": "partial"}]
    _ev(event="vault_land", ok=True, item_id=586, commit="aaa1", review="pass", review_clauses=MET5)
    _ev(event="vault_land", ok=True, item_id=586, commit="bbb2", review="pass", review_clauses=partial)
    _ev(event="vault_land", ok=True, item_id=586, commit="ccc3", review="skipped", review_clauses=[])
    assert B.vault_review_outcome(S.LEDGER_PATH, ["aaa1", "ccc3"]) is None, "ungraded newest"
    assert B.vault_review_outcome(S.LEDGER_PATH, ["aaa1", "bbb2"]) is None, "short newest"
    got = B.vault_review_outcome(S.LEDGER_PATH, ["bbb2", "aaa1"])
    assert got["acceptance"] == "met" and got["source"] == "vault_review"
    assert [c["clause"] for c in got["clause_outcomes"]] == [1, 2, 3, 4, 5]


def test_a_vault_landing_then_the_wall_clock_still_closes(isolated, monkeypatch):
    """The `TurnTimeout` exit wrote its `finished` row with no `vault_commits`
    and no `surface`, so the same landing was invisible by that exit."""
    p = write_item(isolated, 598, clauses=["a"])
    _ev(event="backlog_triage", item_id=598, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=["a"])
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))

    async def fake(prompt, **kw):
        _ev(event="vault_land", ok=True, item_id=598, commit="598a0000", review="pass",
            review_clauses=[{"clause": 1, "verdict": "met"}])
        raise C.TurnTimeout("turn outlived its 3540 s budget")
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(I.execute(type("Q", (), {"payload": {}})()))
    fin = [e for e in _events("backlog_implement") if e.get("phase") == "finished"][-1]
    assert fin["vault_commits"] == ["598a0000"] and fin["surface"] == "vault"
    assert _fm(p)["status"] == "done"


def test_overlapping_sweeps_record_a_landing_once(isolated):
    """Housekeeping sweeps in a worker thread while a vault turn's end sweeps
    on the event loop; the marker check and the write are not atomic."""
    import threading
    write_item(isolated, 599)
    _vault_turn(599)
    results: list = []
    threads = [threading.Thread(target=lambda: results.append(B.close_settled_items(S.LEDGER_PATH)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(len(r) for r in results) == 1
    assert len(_events("item_landed")) == 1


# ── the grader records what it graded ───────────────────────────────────────

def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    r = tmp_path / "obsidian"
    (r / "skills" / "foo").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(r))
    _git(r, "config", "user.email", "t@e.com"); _git(r, "config", "user.name", "t")
    (r / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo\n")
    _git(r, "add", "-A"); _git(r, "commit", "-q", "-m", "base")
    monkeypatch.setattr(V, "VAULT", r)
    monkeypatch.setattr(V, "loader_errors", lambda paths: [])
    return r


def test_a_passing_vault_land_carries_the_graders_clause_verdicts(vault, monkeypatch):
    clauses = [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "met"}]
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("pass", "both met", clauses))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v2\n")
    V.land(["skills/foo/SKILL.md"], "skill: foo v2", item_id=9)
    assert _events("vault_land")[-1]["review_clauses"] == clauses
    assert _events("vault_review")[-1]["clauses"] == clauses


def test_a_two_element_grader_still_lands_and_grades_nothing(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("pass", "all clauses met"))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v3\n")
    assert V.land(["skills/foo/SKILL.md"], "skill: foo v3", item_id=9)["review"] == "pass"
    assert _events("vault_land")[-1]["review_clauses"] == []


def test_a_skipped_review_records_no_verdicts_even_if_handed_some(vault, monkeypatch):
    monkeypatch.setattr(V, "GRADER", lambda **kw: ("skipped", "grader down", MET5))
    (vault / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\n---\n# foo v4\n")
    V.land(["skills/foo/SKILL.md"], "skill: foo v4", item_id=9)
    assert _events("vault_land")[-1]["review_clauses"] == []


def test_grade_vault_returns_one_row_per_contract_clause(isolated, monkeypatch, tmp_path):
    """After `parse_review`'s downgrades — a `met` whose evidence is not on disk
    reads `partial`, and so does a clause the grader itself judged `partial` —
    one row per clause, so a reader counts against the contract as it stood when
    it was graded.

    Re-pointed by #2263. The fixture used to answer TWO of three clauses and
    assert the third came back as a synthesized `partial`, which is what a clause
    the grader never reached used to look like: a verdict on the change, on a
    clause nobody read, that read as a `retry` refusal and spent the item's two
    attempts. The census of those refusals is the 9 `vault_review` rows reading
    "not addressed by the grader". A skipped clause is a grader failure now —
    `tests/test_automod_review.py::test_a_merged_grading_that_leaves_a_clause_unanswered_is_no_verdict`
    pins that — and what this node keeps pinning is that the grader's own `met`
    does not survive a file that is not there."""
    write_item(isolated, 590, clauses=["a", "b", "c"])
    _ev(event="backlog_triage", item_id=590, verdict="confirmed", surface="vault",
        acceptance="a", acceptance_clauses=["a", "b", "c"])
    (tmp_path / "skills" / "x").mkdir(parents=True)
    (tmp_path / "skills" / "x" / "SKILL.md").write_text("x")
    monkeypatch.setattr(RV, "run_grader", lambda **kw: {"ok": True, "structured": {
        "premise": "sound", "summary": "ok", "test_honesty": [], "seams_unverified": [],
        "clauses": [
            {"clause": 1, "verdict": "met", "evidence_path": "skills/x/SKILL.md",
             "evidence_line": 1, "test_node_id": "", "how_verified": "read", "note": ""},
            {"clause": 2, "verdict": "met", "evidence_path": "skills/nope.md",
             "evidence_line": 1, "test_node_id": "", "how_verified": "read", "note": ""},
            {"clause": 3, "verdict": "partial", "evidence_path": "skills/x/SKILL.md",
             "evidence_line": 1, "test_node_id": "", "how_verified": "read", "note": ""}]}})
    kind, _, clauses = RV.grade_vault(item_id=590, paths=["skills/x/SKILL.md"], diff="+x",
                                      vault=tmp_path)
    assert clauses == [{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "partial"},
                       {"clause": 3, "verdict": "partial"}]
    assert kind == "retry"


def test_grade_vault_skips_carry_no_verdicts(isolated, monkeypatch):
    _ev(event="backlog_triage", item_id=591, verdict="confirmed", surface="code",
        acceptance="a", acceptance_clauses=["a"])
    monkeypatch.setattr(RV, "run_grader", lambda **kw: pytest.fail("grader must not run"))
    assert RV.grade_vault(item_id=591, paths=["skills/x/SKILL.md"], diff="+x")[2] == []


# ── the implementer is told what a vault item is ────────────────────────────

def _render(surface):
    return I.PROMPT.format(
        item_id=575, status="up_next", priority="medium", name="n", body="b",
        triaged_ago="today", surface=surface, check="c", evidence="e", acceptance="a",
        clauses="    1. a", spawn_cap=I.SPAWN_CAP, max_turns=I.DEFAULT_MAX_TURNS,
        gate_minutes=16, first_gate_by=23, reoffer="", members="", human_clauses="",
        surface_rules=I._surface_rules(surface, 575))


def test_a_vault_item_is_told_not_to_open_a_round():
    text = " ".join(_render("vault").split())
    assert "a vault change is not a round" in text
    assert "`automod_vault_land(paths, message, item_id=575)`" in text
    assert "do not open a code round to add tests" in text
    assert "asks for no test in `~/lloyd`" in text
    # Read before the code-round lines it overrides.
    assert text.index("a vault change is not a round") < text.index("a test pins each")


@pytest.mark.parametrize("surface", ["code", "frontend", "mixed", ""])
def test_every_other_surface_reads_exactly_as_before(surface):
    assert I._surface_rules(surface, 575) == ""
    assert "a vault change is not a round" not in _render(surface)


def test_the_turn_end_closes_a_vault_landing_before_the_reconcile(isolated, monkeypatch):
    """#575's two minutes, end to end through `execute`: the turn opens a
    round, lands the vault fix with a passing review, and dies at its budget.
    Without the sweep in `execute`'s `finally`, the reconcile moved the item
    to `up_next` and `select_confirmed` handed it to the next round."""
    p = write_item(isolated, 575, clauses=["a", "b"])
    _ev(event="backlog_triage", item_id=575, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a; b", acceptance_clauses=["a", "b"])
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))

    async def fake(prompt, **kw):
        fake.prompt = prompt
        _ev(event="round_start", round_id="SM_20260914_212024", item_id=575)
        _ev(event="vault_land", ok=True, item_id=575, commit="588339ab0000", review="pass",
            review_clauses=[{"clause": 1, "verdict": "met"}, {"clause": 2, "verdict": "met"}])
        return {"text": "", "session_id": "s", "stop_reason": "max_turns", "num_turns": 151,
                "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(I.execute(type("Q", (), {"payload": {}})()))
    assert "a vault change is not a round" in " ".join(fake.prompt.split())
    assert _fm(p)["status"] == "done"
    assert B.select_confirmed(S.LEDGER_PATH) is None, "nothing left for the next round to take"


def test_a_turn_with_no_vault_landing_does_not_pay_for_the_sweep(isolated, monkeypatch):
    """The sweep walks the whole board on the event loop (~2.3 s measured).
    A code landing waits out the guardian's window, which housekeeping covers,
    so only a turn that landed in the vault needs it at its end."""
    write_item(isolated, 595, clauses=["a"])
    _ev(event="backlog_triage", item_id=595, verdict="confirmed", surface="code",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=["a"])
    # A vault landing for ANOTHER item during this turn does not count either.
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    monkeypatch.setattr(B, "close_settled_items",
                        lambda *a, **k: pytest.fail("no vault landing this turn; no sweep"))

    async def fake(prompt, **kw):
        _ev(event="vault_land", ok=True, item_id=999, commit="0ther000", review="pass",
            review_clauses=[{"clause": 1, "verdict": "met"}])
        return {"text": "", "session_id": "s", "stop_reason": "max_turns", "num_turns": 151,
                "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(I.execute(type("Q", (), {"payload": {}})()))
    assert _fm(isolated / "595-a-thing.md")["status"] == "up_next", "re-offered as before"


# ── "it comes back once" ────────────────────────────────────────────────────

def _budget_death(item_id, rid):
    _ev(event="backlog_implement", item_id=item_id, phase="started")
    _ev(event="backlog_implement", item_id=item_id, phase="finished", round_id=rid,
        stop_reason="max_turns", num_turns=151)


def test_a_budget_death_is_re_offered_once(isolated):
    _budget_death(592, "SM_A")
    assert B.implement_outcomes(S.LEDGER_PATH)[592][0] == "incomplete"
    _budget_death(592, "SM_B")
    assert B.implement_outcomes(S.LEDGER_PATH)[592][0] == "spent", \
        "the second budget death is not re-offered"


def test_an_earlier_different_verdict_does_not_spend_the_re_offer(isolated):
    """Counting all attempts would: an external block, then a budget death,
    would read `n = 2` and spend the item on the first time it ran out."""
    _ev(event="backlog_implement", item_id=593, phase="started")
    _ev(event="backlog_implement", item_id=593, phase="finished", round_id="SM_X",
        stop_reason="stop", num_turns=40)
    _ev(event="gate", round_id="SM_X", rung="tests", ok=False, external_blocker=True)
    assert B.implement_outcomes(S.LEDGER_PATH)[593][0] == "external"
    _budget_death(593, "SM_Y")
    assert B.implement_outcomes(S.LEDGER_PATH)[593][0] == "incomplete"


def test_a_human_reopen_restores_the_re_offer(isolated):
    write_item(isolated, 594)
    _budget_death(594, "SM_1"); _budget_death(594, "SM_2")
    assert B.implement_outcomes(S.LEDGER_PATH)[594][0] == "spent"
    B.reopen_item(594, "try again", ledger=S.LEDGER_PATH)
    _budget_death(594, "SM_3")
    assert B.implement_outcomes(S.LEDGER_PATH)[594][0] == "incomplete"


def test_triage_does_not_write_a_test_file_into_a_vault_contract():
    """Every clause ended "— tests/<file>.py", vault items included, so the
    contract itself sent a vault round to write a test — and #575's 22:02Z
    vault review refused partly over a pinning suite that did not exist."""
    from workers.sources import autotriage as M
    single = " ".join(M.PROMPT.split())
    group = " ".join(M.GROUP_PROMPT.split())
    desc = B.TRIAGE_VERDICT_SCHEMA["properties"]["acceptance_clauses"]["description"]
    assert "A `vault` surface is the exception" in single
    assert "the vault path that shows it (`— skills/<name>/SKILL.md`)" in single
    assert "for a `vault` umbrella, the vault path that shows it instead, never a test" in group
    assert "for a `vault` surface the vault path that shows it, never a test" in desc


# ── #2263 clause 5: a vault close cannot outrank the absence of grader rows ──
#
# The failure this pins is not the skipped land, it is what came after it. Item
# 2260: grader row `kind: skipped`, `clauses: []`, `blocking: false` at
# 2026-10-06T01:31:39Z; six minutes later its `backlog_implement` row recorded
# `{'landed': True, 'acceptance': 'met', 'clause_outcomes': []}` and the item went
# to `status: done`. Item 2325 on 2026-10-07T03:06:21Z: `acceptance: met` with
# five SELF-REPORTED `clause_outcomes` beside a grader row of `skipped`/`[]`.
# #2240's comment claimed `backlog.py` "refuses to judge them and the item stays
# open" — that half was true of `vault_review_outcome` and untrue of the close.


def _ungraded_vault_landing(item_id: int, clause_outcomes=(), surface="vault"):
    """The row shape both items left behind: a vault land with no clause verdicts
    and an implement report that says `met` anyway."""
    _ev(event="vault_land", ok=True, commit="aaa1", item_id=item_id,
        paths=["skills/x/SKILL.md"], review="skipped",
        review_reason="grader did not answer: generation diverged at 8192 tokens",
        landing_clauses=[])
    _ev(event="backlog_implement", phase="finished", item_id=item_id, round_id="",
        surface=surface, vault_commits=["aaa1"],
        outcome={"landed": True, "acceptance": "met", "clause_outcomes": list(clause_outcomes),
                 "deferred_to": [], "spawned": [], "summary": "the ruling is in the skill"})


def test_a_vault_landing_with_no_grader_clauses_cannot_report_acceptance_met(isolated):
    """#2260 verbatim: `acceptance: met`, `clause_outcomes: []`, no grader row.
    On the vault surface there is no code review rung behind a no-round landing,
    so the grader's rows are the ONLY per-clause evidence — a `met` with none is
    the turn grading its own homework."""
    write_item(isolated, 900, clauses=["the ruling is in the skill", "the witness is on disk",
                                       "no interval job is scheduled"])
    _ev(event="backlog_triage", item_id=900, verdict="confirmed", surface="vault",
        acceptance="the ruling is on disk", acceptance_clauses=["the ruling is in the skill",
                                                                "the witness is on disk",
                                                                "no interval job is scheduled"])
    _ungraded_vault_landing(900)
    landed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}
    out = landed[900]["outcome"]
    assert out["acceptance"] == "not_met", f"a met with zero grader rows: {out}"
    assert out["clause_outcomes"] == []
    assert out["acceptance_ungraded"], "the close must say it was UNGRADED, not refused"
    assert out["landed"] is True and out["summary"] == "the ruling is in the skill"
    got = {g["item_id"]: g for g in B.close_settled_items(S.LEDGER_PATH, dry_run=True)}
    assert got[900]["closed"] is False, got[900]
    assert "never graded" in got[900]["reason"], got[900]["reason"]


def test_self_reported_clause_rows_do_not_satisfy_a_contract_the_grader_skipped(isolated):
    """#2325's shape, which is the worse one: five `clause_outcomes` the turn wrote
    itself, beside a grader row that answered nothing. What is on the row is not
    evidence — the reader that counts clauses reads the grader's."""
    write_item(isolated, 901, clauses=[f"clause {i}" for i in range(1, 6)])
    _ev(event="backlog_triage", item_id=901, verdict="confirmed", surface="vault",
        acceptance="x", acceptance_clauses=[f"clause {i}" for i in range(1, 6)])
    _ungraded_vault_landing(901, clause_outcomes=[
        {"clause": i, "outcome": "met", "evidence": "skills/x/SKILL.md", "test_node_id": "",
         "how_verified": "read", "note": f"self-reported {i}"} for i in range(1, 6)])
    landed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}
    assert landed[901]["outcome"]["acceptance"] == "not_met", landed[901]["outcome"]
    assert landed[901]["outcome"]["clause_outcomes"] == [], \
        "the turn's own rows must not ride along as the grader's grading"


def test_a_grader_that_answered_per_clause_still_closes_the_vault_item(isolated):
    """The boundary, from the other side: the same item with a `pass` review
    carrying a row per clause, matched by commit, is a graded `met` and closes.
    Without this the check above would just be a refusal to land vault items."""
    write_item(isolated, 902, clauses=["a", "b", "c"])
    _ev(event="backlog_triage", item_id=902, verdict="confirmed", surface="vault",
        acceptance="a;b;c", acceptance_clauses=["a", "b", "c"])
    _ev(event="vault_review", item_id=902, kind="pass", blocking=False,
        clauses=[{"clause": i, "verdict": "met"} for i in range(1, 4)])
    _ev(event="vault_land", ok=True, commit="bbb2", item_id=902,
        paths=["skills/x/SK.md"], review="pass",
        # `land()` writes the grader's rows and the landing's own onto the LAND row
        # under `review_clauses`, and that is what `vault_review_outcome` counts.
        review_clauses=[{"clause": i, "verdict": "met"} for i in range(1, 4)],
        landing_clauses=[{"clause": 3, "verdict": "met", "commit": "bbb2"}])
    _ev(event="backlog_implement", phase="finished", item_id=902, round_id="",
        surface="vault", vault_commits=["bbb2"],
        outcome={"landed": True, "acceptance": "met", "clause_outcomes": [],
                 "deferred_to": [], "spawned": [], "summary": "all three"})
    landed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}
    assert landed[902]["outcome"]["acceptance"] == "met", landed[902]["outcome"]
    got = {g["item_id"]: g for g in B.close_settled_items(S.LEDGER_PATH)}
    assert got[902]["closed"] is True, got[902]


def test_a_mixed_surface_landing_is_left_to_the_code_rung(isolated):
    """#551's boundary, unchanged: a `mixed` item's clauses belong to the code
    gate, so its vault half is never going to carry a vault verdict and the close
    must not wait for one. The check is scoped to the surface the grader answers."""
    write_item(isolated, 903, clauses=["the skill names it", "tests/test_x.py::t passes"])
    _ev(event="backlog_triage", item_id=903, verdict="confirmed", surface="mixed",
        acceptance="x", acceptance_clauses=["the skill names it", "tests/test_x.py::t passes"])
    _ungraded_vault_landing(903, surface="mixed")
    landed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}
    assert landed[903]["outcome"]["acceptance"] == "met", "the mixed half is not ungraded"
    got = {g["item_id"]: g for g in B.close_settled_items(S.LEDGER_PATH)}
    assert got[903]["closed"] is True, \
        "closing on the turn's word is wrong for a vault item and right for this one"


# ── #2691: the land refused because HEAD already held every path it named ───
#
# A concurrent job's no-pathspec pre-flight sweep commits the round's in-flight
# bytes; the author's `automod_vault_land` then correctly refuses and names the
# taking sha in `held_by` (#2175's fix `249a2f5e`). What #2175 did not do is
# reach the consumer: `git log -S'held_by' -- scripts/automod/ workers/sources/`
# names that one commit and no reader. So #2684's review graded all five clauses
# `met` at 07:14:03Z, its content had been on `main` since 07:12:26Z, and at
# 07:27:42Z the item was moved to `draft` with "its one unattended attempt is
# spent" and re-triaged. Measured over the whole ledger: 3 finished rows in that
# shape (#2287, #2649, #2684) and all 3 moved to `draft`; only #2684's review was
# an all-`met` pass, so only that one was a wrong close. `held_by_head_landing`
# is the reader, and `settled_landings` and `autocode._vault_landed_since` both
# go through it.

#: #2684's taking sha, from its `vault_land` row: `autonomy-data-pipeline:
#: pre-flight 2026-10-14 (unattributed dirty state)` at 07:12:26Z, whose body
#: lists this path among the 13 it says it did not author.
TAKEN_SHA = "b0248328"
TAKEN_PATH = "skills/retrieval-eval/SKILL.md"

#: The exact reason `reconcile_statuses` wrote for #2684 at 07:27:42Z.
SPENT_REASON = "its one unattended attempt is spent"

CLAUSES5 = [f"clause {i}" for i in range(1, 6)]


def _held_by_head_turn(item_id, *, paths=(TAKEN_PATH,), held_by=None, surface="vault",
                       review=("pass", MET5), outcome=None, session_id="s",
                       shortfall=""):
    """#2684's three rows in its own order: the vault review that graded the
    contract, the land that refused with `held_by` because HEAD already held
    every path it named (`ok: false`, no commit, `review: "skipped"` — the
    grader does not run on a refusal), and the finished turn whose
    `vault_commits` is empty and which opened no round.

    `outcome` stayed unused at `None` for every node written against this helper
    until #3186, and that default is the whole reason the bug survived its tests:
    a refused land raises, so a real turn's finished row always reports
    `acceptance: "not_met"`. #3186's nodes below are the first to hand it that
    pairing."""
    kind, clauses = review
    _ev(event="vault_review", item_id=item_id, kind=kind, blocking=False,
        verdict_shortfall=shortfall, paths=list(paths), clauses=list(clauses))
    _ev(event="vault_land", ok=False, item_id=item_id, commit=None, review="skipped",
        review_reason="the grader does not run on a refusal", paths=list(paths),
        held_by=dict(held_by if held_by is not None else {p: TAKEN_SHA for p in paths}))
    _ev(event="backlog_implement", item_id=item_id, phase="finished", round_id=None,
        surface=surface, vault_commits=[], session_id=session_id, outcome=outcome)


def _triaged(d: Path, item_id, clauses=CLAUSES5, surface="vault"):
    write_item(d, item_id, clauses=list(clauses))
    _ev(event="backlog_triage", item_id=item_id, verdict="confirmed", surface=surface,
        check="ls", evidence="e", acceptance="; ".join(clauses), acceptance_clauses=list(clauses))


# ── clause 1: settled_landings reports it, as a landing of its own kind ─────

def test_a_held_by_head_refusal_is_a_landing_with_its_own_kind(isolated):
    """#2684 verbatim, read by `settled_landings`: a landing whose commit is not
    the turn's, reported with a `kind` a committed landing does not carry.

    The grading comes from the `vault_review` event, and the reason it has to is
    asserted here too: `vault_review_outcome` reads per-clause rows off an **ok**
    land row keyed by commit, so on this shape — no commit, `review: "skipped"` —
    it returns None and the all-`met` grading would be invisible."""
    _triaged(isolated, 2684)
    _held_by_head_turn(2684)
    assert B.vault_review_outcome(S.LEDGER_PATH, []) is None, \
        "the old reader cannot see this grading: no commit to key on"
    landed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}
    assert 2684 in landed, "the refusal is a landing now"
    row = landed[2684]
    assert row["kind"] == B.HELD_BY_HEAD == "held_by_head"
    assert row["commit"] == TAKEN_SHA, "the taking sha, which is the content on main"
    assert row["round_id"] == "" and row["vault"] is True
    assert row["outcome"]["acceptance"] == "met"
    assert [c["clause"] for c in row["outcome"]["clause_outcomes"]] == [1, 2, 3, 4, 5]
    # A committed landing of the same pass carries no `kind` at all, which is
    # what makes this one tellable.
    _ev(event="vault_review", item_id=902, kind="pass", blocking=False,
        clauses=[{"clause": 1, "verdict": "met"}])
    _ev(event="vault_land", ok=True, commit="bbb2", item_id=902, paths=["skills/x/SK.md"],
        review="pass", review_clauses=[{"clause": 1, "verdict": "met"}],
        landing_clauses=[{"clause": 1, "verdict": "met", "commit": "bbb2"}])
    _ev(event="backlog_implement", phase="finished", item_id=902, round_id="",
        surface="vault", vault_commits=["bbb2"],
        outcome={"landed": True, "acceptance": "met", "clause_outcomes": [],
                 "deferred_to": [], "spawned": [], "summary": "it landed"})
    committed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}[902]
    assert "kind" not in committed, committed


def test_the_close_names_the_taking_sha_and_no_draft_row_is_written(isolated):
    """The other half of #2684: the item is closed, the sha is in the marker and
    in the ledger row and in the note, and the `draft` row that cost it a second
    and third triage is never written."""
    p = write_item(isolated, 2684, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=2684, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _held_by_head_turn(2684)
    got = {g["item_id"]: g for g in B.close_settled_items(S.LEDGER_PATH)}
    assert got[2684]["closed"] is True, got[2684]
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == TAKEN_SHA
    ev = _events("item_landed")[-1]
    assert ev["commit"] == TAKEN_SHA and ev["kind"] == "held_by_head", ev
    assert TAKEN_SHA in fm["activity_log"][-1], "the note says where the content came from"
    moved = [m for m in _events("status_moved") if m.get("to") == "draft"]
    assert moved == [], f"a held-by-HEAD close must not write {SPENT_REASON!r}: {moved}"
    B.reconcile_statuses(S.LEDGER_PATH)
    assert _fm(p)["status"] == "done", "the reconcile leaves a closed item closed"
    assert [m for m in _events("status_moved") if m.get("to") == "draft"] == []


# ── clause 3: the implement route itself, and its counterfactual ────────────

def _run_held_by_head_turn(d: Path, monkeypatch, item_id, clauses=("a", "b")):
    """`execute` over #2684's turn: the triage, the all-`met` review, the
    `held_by` refusal, and a turn that ends with nothing to report. Its own
    end-of-turn sweep is the route under test."""
    p = write_item(d, item_id, clauses=list(clauses))
    _ev(event="backlog_triage", item_id=item_id, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="; ".join(clauses),
        acceptance_clauses=list(clauses))
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))

    async def fake(prompt, **kw):
        _ev(event="vault_review", item_id=item_id, kind="pass", blocking=False,
            verdict_shortfall="", paths=[TAKEN_PATH],
            clauses=[{"clause": i, "verdict": "met"} for i in range(1, len(clauses) + 1)])
        _ev(event="vault_land", ok=False, item_id=item_id, commit=None, review="skipped",
            paths=[TAKEN_PATH], held_by={TAKEN_PATH: TAKEN_SHA})
        return {"text": "", "session_id": "s", "stop_reason": "stop", "num_turns": 48,
                "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(I.execute(type("Q", (), {"payload": {}})()))
    return p


def test_the_turn_end_closes_a_held_by_head_landing_before_the_reconcile(isolated, monkeypatch):
    """#2684's 07:27:40Z→07:27:42Z, end to end through `execute`. The turn ends
    with `vault_commits: []` and no round, so the gate that decides whether to
    sweep at all is the one that had to learn this shape; without it the
    reconcile wrote the spent-draft row two seconds later."""
    p = _run_held_by_head_turn(isolated, monkeypatch, 2684)
    fin = [e for e in _events("backlog_implement") if e.get("phase") == "finished"][-1]
    assert fin["vault_commits"] == [] and not fin["round_id"] and fin["surface"] == "vault"
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == TAKEN_SHA
    assert [m for m in _events("status_moved") if m.get("to") == "draft"] == []
    assert B.select_confirmed(S.LEDGER_PATH) is None, "nothing left for the next round to take"


def test_the_same_rows_with_the_close_suppressed_still_spend_the_item(isolated, monkeypatch):
    """The counterfactual clause 3 asks for: same rows, predicate forced to
    abstain, and the exact `draft` row #2684 got is written again. Without this
    node the node above would pass for any reason at all."""
    monkeypatch.setattr(B, "held_by_head_landing", lambda *a, **k: None)
    p = _run_held_by_head_turn(isolated, monkeypatch, 2684)
    fm = _fm(p)
    assert fm["status"] == "draft" and "automod_landed" not in fm
    moved = [m for m in _events("status_moved") if m.get("to") == "draft"]
    assert len(moved) == 1 and SPENT_REASON in moved[0]["reason"], moved


def test_both_routes_read_the_property_through_one_predicate(isolated, monkeypatch):
    """The #575 lesson, stated as a shape: one property, one predicate. The
    turn-end gate must not keep its own copy of the rule, because two readers of
    one property is what re-offered a landed item two minutes after it landed."""
    seen = []
    real = B.held_by_head_landing

    def spy(rows, item_id, **kw):
        seen.append(item_id)
        return real(rows, item_id, **kw)

    monkeypatch.setattr(B, "held_by_head_landing", spy)
    assert I._vault_landed_since(2684, 0.0) is False, "no rows at all: no landing"
    _held_by_head_turn(2684)
    assert I._vault_landed_since(2684, 0.0, session_id="s") is True
    assert 2684 in seen, "the turn-end gate asked the shared predicate"


# ── clauses 4 and 5: the negatives that keep this from being a rubber stamp ─

@pytest.mark.parametrize("kind,clauses,label", [
    ("retry", MET5[:4] + [{"clause": 5, "verdict": "partial"}],
     "#2287/#2649: a `retry` carrying a `partial`"),
    ("diverged", MET5[:2], "#2649's first row: not a pass at all"),
    ("skipped", [], "a grader that answered nothing: no clause rows to count"),
], ids=["retry-partial", "diverged", "skipped-empty"])
def test_a_review_that_is_not_an_all_met_pass_still_spends_the_item(isolated, kind, clauses, label):
    """The boundary the close must not cross. Both of the other two ledger items
    in this shape (#2287, #2649) had non-all-met reviews, and their move to
    `draft` was correct — this branch stays out of their way."""
    write_item(isolated, 2287, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=2287, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _held_by_head_turn(2287, review=(kind, clauses))
    assert B.held_by_head_landing(_events("vault_land") + _events("vault_review"),
                                  2287, session_id="s") is None, label
    assert {d["item_id"] for d in B.settled_landings(S.LEDGER_PATH)} == set(), label
    assert B.close_settled_items(S.LEDGER_PATH) == []
    assert B.implement_outcomes(S.LEDGER_PATH)[2287][0] == "spent", label
    want = {int(k): v for k, v in B.desired_statuses(S.LEDGER_PATH).items()}
    assert want[2287][0] == "draft" and SPENT_REASON in want[2287][1]


def test_a_refusal_that_covers_only_some_named_paths_closes_nothing(isolated):
    """#2175 clause 4's producer-side rule, read on the consumer side:
    `_held_by_head` yields no entry for a path git never committed, so a partially
    covered refusal is not proof HEAD holds the round's work. Nothing is closed
    and no sha is recorded — the item stays exactly where it is today."""
    p = write_item(isolated, 2692, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=2692, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _held_by_head_turn(2692, paths=[TAKEN_PATH, "knowledge/new-note.md"],
                       held_by={TAKEN_PATH: TAKEN_SHA})
    assert B.held_by_head_landing(_events("vault_land") + _events("vault_review"),
                                  2692, session_id="s") is None
    assert {d["item_id"] for d in B.settled_landings(S.LEDGER_PATH)} == set()
    assert B.close_settled_items(S.LEDGER_PATH) == []
    fm = _fm(p)
    assert "automod_landed" not in fm and TAKEN_SHA not in str(fm)
    assert fm["status"] == "up_next", "left where it is today, not closed on half a witness"


def test_a_self_healed_round_is_left_to_the_normal_closer(isolated):
    """17 of the 20 refusals that name a taking sha self-healed because another
    named path landed `ok: true` in the same turn, so `vault_commits` was non-empty
    and the ordinary route ran. This branch fires only when EVERY named path is
    held, and stays out when one landed."""
    write_item(isolated, 2287, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=2287, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _held_by_head_turn(2287)
    _ev(event="vault_land", ok=True, item_id=2287, commit="cc01", review="pass",
        ts=(_events("vault_land")[-1]["ts"] or 0) + 1, paths=["skills/other/SKILL.md"],
        review_clauses=MET5)
    assert B.held_by_head_landing(_events("vault_land") + _events("vault_review"),
                                  2287, session_id="s") is None, \
        "a committed landing beside it is the normal reader's to see"


def test_another_items_refusal_is_never_borrowed(isolated):
    """`item_id` is optional on the `automod_vault_land` tool, so a continued
    session carries rows that name no item at all (189 of the 870 `vault_land`
    rows in `promotions.jsonl` and its two monthly archives on 2026-10-17, 133 of
    those with a session) beside rows that name one. A row naming SOMEONE ELSE's
    item must not be matched by session — that would buy a close for content this
    item never wrote — while a row naming no item is still matched by session, as
    `_vault_commits_since` has always done."""
    write_item(isolated, 2693, clauses=["a"])
    _ev(event="backlog_triage", item_id=2693, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=["a"])
    _held_by_head_turn(999)
    rows = _events("vault_land") + _events("vault_review")
    assert B.held_by_head_landing(rows, 2693, session_id="s") is None, \
        "#999 named itself; the session does not make it #2693's"
    _ev(event="vault_review", kind="pass", blocking=False, verdict_shortfall="",
        session_id="s", paths=[TAKEN_PATH], clauses=[{"clause": 1, "verdict": "met"}])
    _ev(event="vault_land", ok=False, commit=None, review="skipped", session_id="s",
        paths=[TAKEN_PATH], held_by={TAKEN_PATH: TAKEN_SHA})
    rows = _events("vault_land") + _events("vault_review")
    got = B.held_by_head_landing(rows, 2693, session_id="s")
    assert got and got["commit"] == TAKEN_SHA, "a row naming no item is the session's"


def test_the_short_sha_held_by_carries_is_recorded_as_it_stands(isolated):
    """`_held_by_head` asks git with `--format=%h`, so the sha this route records is
    the SHORT form: #2684's refusal row carries `b0248328`, eight characters, where
    `git rev-parse` prints forty. It is recorded exactly as the row carried it — the
    marker, the closing row and the note all hold the same string, and the item is
    still landed afterwards, not re-read as an attempt spent because the sha came
    in short."""
    p = write_item(isolated, 2684, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=2684, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _held_by_head_turn(2684)
    assert len(TAKEN_SHA) == 8, "the fixture is the shape git actually returns here"
    got = {g["item_id"]: g for g in B.close_settled_items(S.LEDGER_PATH)}
    assert got[2684]["closed"] is True, got[2684]
    fm = _fm(p)
    assert fm["automod_landed"] == TAKEN_SHA
    assert _events("item_landed")[-1]["commit"] == TAKEN_SHA
    B.reconcile_statuses(S.LEDGER_PATH)
    assert _fm(p)["status"] == "done", "a short sha is still a landing to the reconcile"
    assert [m for m in _events("status_moved") if m.get("to") == "draft"] == []


# ── #3186: the refusal is also the word that silences the rescue for it ──────
#
# The clean-index branch raises (`vault_round.py`'s `_no_change_refusal`), so the
# finalizer reading that raise has no acceptance word available but `not_met`. The
# gate in `settled_landings` treated ANY usable non-`met` report as #575's
# substantiated refusal and skipped the #2691 rescue — so the report the refusal
# itself produced was the thing that silenced the route built for that refusal.
# Census over `promotions.jsonl` and its two monthly archives: 53 `vault_land` rows
# whose error opens `nothing to commit on those paths`, all 53 with a non-empty
# `held_by`, 24 of them naming a `pre-flight`/`pre-run snapshot` commit — against 3
# `kind: "held_by_head"` rows in the whole ledger, and four refusals on 2026-10-23
# with zero rescues. #3182 is the instance: its review graded all five clauses
# `met` at 07:50:09Z, `64ebdc6a` took both its paths 23 seconds later, its land
# refused at 07:56:50Z, and `status_moved` in_progress → draft went out at
# 07:58:25Z with "its one unattended attempt is spent".

#: #3182's two paths, both taken by the same sweep and so both naming one sha —
#: the shape `held_by_head_landing`'s sha-choice rule was written for.
TAKEN_PATHS = ("backlog/data/2026-10-16.2836-logs-server.log",
               "backlog/data/2026-10-16.2836-logs-server.log.md")

#: #3182's finished-turn `outcome` as its `backlog_implement` row carries it:
#: `acceptance: not_met`, `landed: false`, one row per clause — four `met` and a
#: fifth whose evidence is the refusal itself. `outcome_carries_no_claim` is False
#: on it, which is exactly why the gate read it as a refusal about the work.
LAND_REFUSAL_OUTCOME = {
    "landed": False, "acceptance": "not_met", "deferred_to": [], "spawned": [],
    "summary": "the land refused: nothing to commit on those paths",
    "clause_outcomes": [
        {"clause": i, "outcome": "met", "deferred_to": [],
         "evidence": "the extract is on disk and byte-identical to HEAD's blob"}
        for i in range(1, 5)] + [
        {"clause": 5, "outcome": "not_met", "deferred_to": [],
         "evidence": "automod_vault_land raised: nothing to commit on those paths, "
                     "HEAD already holds them"}]}


# ── clause 1: the refusal's own `not_met` does not unmake the landing ─────────

def test_a_land_refusal_that_forced_the_not_met_report_is_still_a_landing(isolated):
    """#3182's rows read by `settled_landings`, including the outcome its refused
    land forced onto the turn — the pairing no existing node ever handed the
    helper, because a refusal raises and the node author had no raise to copy.

    The landing is still the taking sha, still `kind: held_by_head`, and its
    acceptance comes from the grader that graded the whole contract, not from a
    report whose only evidence is the refusal this branch exists to look past."""
    _triaged(isolated, 3182)
    _held_by_head_turn(3182, paths=TAKEN_PATHS, outcome=LAND_REFUSAL_OUTCOME)
    assert B.outcome_carries_no_claim(LAND_REFUSAL_OUTCOME) is False, \
        "the fixture is a USABLE report: that is what made the gate skip it"
    landed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}
    assert 3182 in landed, "the rescue fires even though the turn reported not_met"
    row = landed[3182]
    assert row["kind"] == B.HELD_BY_HEAD == "held_by_head"
    assert row["commit"] == TAKEN_SHA, "the taking sha, which is the content on main"
    assert row["round_id"] == "" and row["vault"] is True
    assert row["outcome"]["acceptance"] == "met", \
        "the grader's all-met verdict decides; the turn's word is the refusal's echo"
    assert [c["clause"] for c in row["outcome"]["clause_outcomes"]] == [1, 2, 3, 4, 5]


# ── clause 2: settle-then-reconcile closes it, twice, and writes no draft row ─

def test_the_settle_then_reconcile_closes_the_attempt_and_writes_no_draft_row(isolated):
    """The consequence #3182 paid: 07:56:50Z refusal, 07:58:25Z `in_progress →
    draft` with "its one unattended attempt is spent", then a re-triage at
    08:09:25Z. With the rescue un-silenced the item is `done` on the taking sha and
    that row is never written — on the first pass or the second."""
    p = write_item(isolated, 3182, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=3182, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _held_by_head_turn(3182, paths=TAKEN_PATHS, outcome=LAND_REFUSAL_OUTCOME)
    got = {g["item_id"]: g for g in B.close_settled_items(S.LEDGER_PATH)}
    assert got[3182]["closed"] is True, got[3182]
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == TAKEN_SHA
    assert _events("item_landed")[-1]["kind"] == "held_by_head"
    B.reconcile_statuses(S.LEDGER_PATH)
    moved = [m for m in _events("status_moved") if m.get("to") == "draft"]
    assert moved == [], f"a held-by-HEAD close must not spend the attempt: {moved}"
    assert _fm(p)["status"] == "done"

    # Housekeeping runs this every tick, so the second pass over the SAME ledger
    # has to leave a closed item closed and still write no draft row.
    B.close_settled_items(S.LEDGER_PATH)
    B.reconcile_statuses(S.LEDGER_PATH)
    assert _fm(p)["status"] == "done", "the second pass reopened a closed item"
    assert [m for m in _events("status_moved") if m.get("to") == "draft"] == []
    assert B.select_confirmed(S.LEDGER_PATH) is None, "nothing left for a round to take"


# ── clause 3: and only beside a verdict that actually graded the contract ─────

@pytest.mark.parametrize("review,shortfall,label", [
    (("retry", MET5[:4] + [{"clause": 5, "verdict": "partial"}]), "",
     "a `retry` carrying a `partial` (#2287/#2649)"),
    (("pass", MET5), "clause 3 has no verdict row",
     "a `pass` that is not a verdict on this contract"),
], ids=["retry-partial", "verdict-shortfall"])
def test_a_forced_not_met_rescues_only_beside_an_all_met_verdict(isolated, review,
                                                                shortfall, label):
    """The relaxation is bounded by the predicate, not by the report. Same forced
    `not_met`, but the newest review is not an all-`met` verdict on the contract:
    no landing, and the item is still moved to `draft` as spent, exactly as #2287
    and #2649 were — their spending was correct."""
    p = write_item(isolated, 3182, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=3182, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _held_by_head_turn(3182, paths=TAKEN_PATHS, review=review, shortfall=shortfall,
                       outcome=LAND_REFUSAL_OUTCOME)
    assert B.held_by_head_landing(_events("vault_land") + _events("vault_review"),
                                  3182, session_id="s") is None, label
    assert {d["item_id"] for d in B.settled_landings(S.LEDGER_PATH)} == set(), label
    assert B.close_settled_items(S.LEDGER_PATH) == []
    assert B.implement_outcomes(S.LEDGER_PATH)[3182][0] == "spent", label
    want = {int(k): v for k, v in B.desired_statuses(S.LEDGER_PATH).items()}
    assert want[3182][0] == "draft" and SPENT_REASON in want[3182][1], label
    assert _fm(p)["status"] == "up_next", "left where it is today, not closed on a partial"


# ── the boundaries the relaxation must not cross ─────────────────────────────

def test_the_same_report_still_decides_a_committed_vault_landing(isolated):
    """#575's guarantee, on the branch this change does NOT touch. A landing that
    made its own commit is still decided by a `not_met` that names its clauses,
    beside a review that graded all five met — confining the override to the
    held-by-HEAD branch is the whole difference between this and a rubber stamp."""
    p = write_item(isolated, 583, clauses=CLAUSES5)
    _ev(event="backlog_triage", item_id=583, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
    _vault_turn(583, round_id=None, stop_reason="stop", outcome=LAND_REFUSAL_OUTCOME)
    assert B.vault_review_outcome(S.LEDGER_PATH, ["588339ab0000"]), \
        "the review graded all five met; the turn's named refusal still decides"
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 583, "closed": False, "acceptance": "not_met"}]
    assert _fm(p)["status"] == "up_next"


def test_a_report_that_names_ids_or_a_verdict_still_outranks_the_rescue(isolated):
    """The other word a refusal cannot produce. `deferred` names ids the turn waits
    on and `unnecessary`/`rejected` are verdicts on the item, none of them the
    mechanical consequence of a raised land, so all three still silence the rescue
    — `outcome_carries_no_claim`'s own docstring calls overriding a `deferred`
    "the one claim this loop cannot take back"."""
    for word in ("deferred", "unnecessary", "rejected"):
        write_item(isolated, 3184, clauses=CLAUSES5)
        _ev(event="backlog_triage", item_id=3184, verdict="confirmed", surface="vault",
            check="ls", evidence="e", acceptance="a", acceptance_clauses=CLAUSES5)
        _held_by_head_turn(3184, paths=TAKEN_PATHS,
                           outcome={**LAND_REFUSAL_OUTCOME, "acceptance": word,
                                    "deferred_to": [9001] if word == "deferred" else []})
        assert {d["item_id"] for d in B.settled_landings(S.LEDGER_PATH)} == set(), word
        assert B.close_settled_items(S.LEDGER_PATH) == [], word


# ── #2913: the finish-side reader must not borrow another item's vault landing ─
#
# `ed5ace11` (#2903) narrowed `round_landing_rows`, the reader behind the
# `landed` field. The finish-side reader it never touched —
# `workers.sources.autocode._vault_commits_since` — decides two other things
# from the same rows: the `vault_commits` list persisted on the
# `backlog_implement` finished row, which `settled_landings` (backlog.py:3745)
# turns into an `item_landed` row for the item it is reading, and
# `_vault_landed_since`, the turn-end sweep's gate on running that close at all.
# So one turn was checked narrowly for `landed` and loosely for the close.
#
# The loose arm has already fired. Joining every `backlog_implement` finished
# row's `vault_commits` to the `vault_land` rows in `promotions.jsonl` and its
# two monthly archives (measured 2026-10-17): of the 528 finished rows carrying
# `vault_commits`, 503 include a row naming their own item and 24 name no item
# at all — the protected direction — while exactly ONE, item #2039 on session
# `20261001_154803_*`, has landing rows naming ONLY another item (#2027).
# Reusing a session is the ordinary case, not the exception: of the 870
# `vault_land` rows in those files on 2026-10-17, 681 carry an `item_id` spread
# over 344 distinct sessions — about two items to a session — and 189 carry
# none, 133 of those a session.

#: #2039's own session, truncated to the prefix the ledger actually holds.
SESSION_2039 = "20261001_154803"


def _land_row(item_id, *, commit, session=SESSION_2039, ok=True):
    """A successful `vault_land` row in the shape the MCP tool handler writes it:
    `item_id` from the call arguments (optional there), `session_id` stamped by
    the handler from the harness's `_meta` (`agent_mcp/automod.py`), never by the
    caller. `item_id=None` is the 189-row shape that has no attribution but its
    session."""
    return {"event": "vault_land", "ok": ok, "item_id": item_id, "commit": commit,
            "session_id": session, "ts": 1.0, "paths": [f"knowledge/{commit}.md"]}


def test_a_vault_landing_row_naming_other_items_is_not_this_items_vault_commit():
    """Clause 1, at the function the close reads.

    Two successful rows on one session, each naming an item other than the one
    being finalised — #2039's ledger exactly. The turn's own session id is what
    the loose arm vouched on, so it is what the call passes.
    """
    rows = [_land_row(2027, commit="a1a1a1a1"), _land_row(2036, commit="b2b2b2b2")]
    assert I._vault_commits_since(rows, 2039, 0.0, session_id=SESSION_2039) == [], (
        "a session match vouched for two landings that belong to other items")
    # Control: the same call with a row naming THIS item is a landing. The empty
    # answer above is attribution, not a blanket refusal to see a vault landing.
    mine = _land_row(2039, commit="c3c3c3c3")
    assert I._vault_commits_since(rows + [mine], 2039, 0.0,
                                  session_id=SESSION_2039) == ["c3c3c3c3"]


def test_a_vault_landing_row_with_no_item_id_is_still_attributed_by_session():
    """Clause 2: the false-positive direction this must never acquire.

    #982's turn called `automod_vault_land` without the optional `item_id`, so
    its row carries no item at all and the harness-stamped session is the only
    attribution it has. Demoting that row would read a vault-surface turn that
    really landed as having landed nothing — 189 of the 870 rows on 2026-10-17
    are in that shape, 133 of them with a session to go on.
    """
    assert I._vault_commits_since([_land_row(None, commit="d4d4d4d4")],
                                  2039, 0.0, session_id=SESSION_2039) == ["d4d4d4d4"]
    # Controls: a matching session is still required, and a row is never
    # attributed to a turn that has no session of its own to match with.
    assert I._vault_commits_since([_land_row(None, commit="e5e5e5e5",
                                             session="20261001_999999")],
                                  2039, 0.0, session_id=SESSION_2039) == []
    assert I._vault_commits_since([_land_row(None, commit="d4d4d4d4")], 2039, 0.0) == []


def _foreign_landing_turn(item_id: int):
    """#2039's turn as the worker records it: a `vault` item that opened no
    round, a session that landed two OTHER items, and the `vault_commits` the
    finisher actually persists — produced by the same call `_run_and_record`
    makes at autocode.py:2149, not a hand-typed list."""
    _ev(event="backlog_triage", item_id=item_id, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=["a"])
    _ev(event="vault_land", **{k: v for k, v in _land_row(2027, commit="a1a1a1a1").items()
                               if k != "event"})
    _ev(event="vault_land", **{k: v for k, v in _land_row(2036, commit="b2b2b2b2").items()
                               if k != "event"})
    vault_commits = I._vault_commits_since(S.read_events(path=S.LEDGER_PATH), item_id,
                                           0.0, session_id=SESSION_2039)
    _ev(event="backlog_implement", phase="finished", item_id=item_id, round_id="",
        surface="vault", session_id=SESSION_2039, vault_commits=vault_commits,
        outcome={"landed": True, "acceptance": "met", "clause_outcomes": [],
                 "deferred_to": [], "spawned": [], "summary": "landed on the vault"})
    return vault_commits


def test_an_item_whose_session_landed_only_other_items_is_never_closed_on_their_shas(
        isolated):
    """Clause 3: the whole finish side, on the #2039 shape.

    What the loose arm costs is not a wrong `landed` field —
    `reconcile_outcome_landing` already answers that one narrowly — it is the
    close: `settled_landings` turns this row's `vault_commits` into an
    `item_landed` row carrying the item's OWN id and the foreign sha
    (backlog.py:3837-3841), so #2039 went to `done` on #2027's commit.
    """
    p = write_item(isolated, 2039, clauses=["a"])
    vault_commits = _foreign_landing_turn(2039)
    assert vault_commits == [], (
        "the finisher persisted another item's vault shas as this item's landing")
    assert I._vault_landed_since(2039, 0.0, session_id=SESSION_2039) is False, (
        "the turn-end sweep would run the close on landings this item never made")
    assert {d["item_id"] for d in B.settled_landings(S.LEDGER_PATH)} == set(), (
        "settled_landings emitted a landing row for an item whose rows name others")
    assert B.close_settled_items(S.LEDGER_PATH) == []
    fm = _fm(p)
    assert fm["status"] == "up_next" and "automod_landed" not in fm, fm

    # Control, same session and same rows but a landing that names ITS OWN item:
    # the machinery still settles and closes it. What stopped #2039 above is the
    # attribution, not a blanket refusal to close a vault item.
    _ev(event="backlog_triage", item_id=2040, verdict="confirmed", surface="vault",
        check="ls", evidence="e", acceptance="a", acceptance_clauses=["a"])
    own_row = _land_row(2040, commit="f6f6f6f6")
    own_row.pop("event")
    own_row["ts"] = (_events("vault_land")[-1]["ts"] or 0) + 1
    _ev(event="vault_land", **own_row)
    _ev(event="backlog_implement", phase="finished", item_id=2040, round_id="",
        surface="vault", session_id=SESSION_2039, vault_commits=["f6f6f6f6"],
        outcome={"landed": True, "acceptance": "met", "clause_outcomes": [],
                 "deferred_to": [], "spawned": [], "summary": "its own landing"})
    landed = {d["item_id"]: d for d in B.settled_landings(S.LEDGER_PATH)}
    assert landed[2040]["commit"] == "f6f6f6f6" and landed[2040]["vault"] is True, landed
    assert I._vault_landed_since(2040, 0.0, session_id=SESSION_2039) is True
