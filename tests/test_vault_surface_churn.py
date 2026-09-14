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


def test_an_outcome_the_turn_reported_is_never_overridden(isolated):
    p = write_item(isolated, 581)
    _vault_turn(581, round_id=None, stop_reason="stop",
                outcome={"acceptance": "not_met", "landed": True, "deferred_to": [],
                         "summary": "", "spawned": []})
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 581, "closed": False, "acceptance": "not_met"}]
    assert _fm(p)["status"] == "up_next"


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
    reads `partial`, and so does a clause the grader never reached — one row
    per clause, so a reader counts against the contract as it stood when it
    was graded."""
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
        reoffer="", members="", human_clauses="",
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
