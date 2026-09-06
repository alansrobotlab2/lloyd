"""Backlog triage: a stale item is a hypothesis that failed, and that is a win.

The property this file exists to protect is that triage cannot start work from
an unverified premise. A backlog going back to February contains items whose
premise no longer holds, and acting on those produces the worst available
outcome: a confident, tested, gated change that solves a problem nobody has.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from scripts.selfmod import backlog as B
from workers.sources import backlog_selfmod as M


def write_item(tmp_path, item_id, *, status="up_next", days_old=100, body="Do the thing.",
               name="A thing", priority="medium", board="lloyd"):
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": priority, "created": created,
          "board": board, "tags": ["backlog"]}
    path = tmp_path / f"{item_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
        encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def backlog_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    return d


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_only_open_items_are_candidates(backlog_dir, tmp_path):
    write_item(backlog_dir, 1, status="done")
    write_item(backlog_dir, 2, status="up_next")
    write_item(backlog_dir, 3, status="closed")
    assert [i.id for i in B.open_items()] == [2]


def test_oldest_untriaged_is_selected_first(backlog_dir, tmp_path):
    """Age is the best proxy for staleness, and finding out which old items are
    still real is the whole point of the pass."""
    write_item(backlog_dir, 10, days_old=30)
    write_item(backlog_dir, 11, days_old=200)
    write_item(backlog_dir, 12, days_old=90)
    assert B.select_candidate(tmp_path / "none.jsonl").id == 11


def test_priority_does_not_override_age(backlog_dir, tmp_path):
    write_item(backlog_dir, 20, days_old=10, priority="high")
    write_item(backlog_dir, 21, days_old=300, priority="low")
    assert B.select_candidate(tmp_path / "none.jsonl").id == 21


def test_an_already_triaged_item_is_not_reselected(backlog_dir, tmp_path):
    write_item(backlog_dir, 30, days_old=300)
    write_item(backlog_dir, 31, days_old=200)
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps(
        {"event": "backlog_triage", "item_id": 30, "verdict": "stale"}) + "\n")
    assert B.select_candidate(ledger).id == 31


def test_nothing_left_to_triage_returns_none(backlog_dir, tmp_path):
    write_item(backlog_dir, 40)
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(json.dumps(
        {"event": "backlog_triage", "item_id": 40, "verdict": "confirmed"}) + "\n")
    assert B.select_candidate(ledger) is None


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def test_a_retiring_verdict_closes_the_item(backlog_dir, tmp_path):
    path = write_item(backlog_dir, 50)
    item = B.load_item(path)
    B.record_verdict(item, "stale", "The module was deleted in abc1234.",
                     check="grep -n foo app/x.py", close=True)
    fm = yaml.safe_load(path.read_text().split("---\n")[1])
    assert fm["status"] == "done"
    assert fm["selfmod_retired"] == "stale"


def test_a_confirmed_verdict_leaves_the_item_open(backlog_dir, tmp_path):
    """Confirmed means there is real work — not that the work is finished."""
    path = write_item(backlog_dir, 51)
    item = B.load_item(path)
    B.record_verdict(item, "confirmed", "Still reproduces on HEAD.", close=True)
    fm = yaml.safe_load(path.read_text().split("---\n")[1])
    assert fm["status"] == "up_next"
    assert "selfmod_retired" not in fm


def test_the_evidence_is_always_written_not_just_the_conclusion(backlog_dir):
    """An item closed with no stated reason is indistinguishable from one closed
    by mistake, and auditability is the whole value of the pass."""
    path = write_item(backlog_dir, 52)
    item = B.load_item(path)
    B.record_verdict(item, "stale", "No caller remains; removed in abc1234.",
                     check="rg -n 'old_fn' app/", close=True)
    text = path.read_text()
    assert "No caller remains" in text
    assert "rg -n 'old_fn' app/" in text
    assert "Selfmod triage" in text
    fm = yaml.safe_load(text.split("---\n")[1])
    assert any("stale" in e for e in fm["activity_log"])


def test_an_unknown_verdict_is_refused(backlog_dir):
    path = write_item(backlog_dir, 53)
    item = B.load_item(path)
    with pytest.raises(ValueError):
        B.record_verdict(item, "probably_fine", "vibes")


def test_the_original_body_survives_annotation(backlog_dir):
    path = write_item(backlog_dir, 54, body="Original description worth keeping.")
    item = B.load_item(path)
    B.record_verdict(item, "confirmed", "Reproduced.")
    assert "Original description worth keeping." in path.read_text()


# ---------------------------------------------------------------------------
# Verdict-block parsing
# ---------------------------------------------------------------------------

def test_a_well_formed_block_parses():
    parsed = M.parse_verdict(
        "prose...\nVERDICT: already_done\nCHECK: pytest tests/test_x.py\n"
        "EVIDENCE: Fixed in 90aa609; the test now passes.\nACCEPTANCE: -")
    assert parsed["verdict"] == "already_done"
    assert parsed["check"] == "pytest tests/test_x.py"


def test_the_last_block_wins_if_the_model_restates_itself():
    parsed = M.parse_verdict(
        "VERDICT: confirmed\nCHECK: a\nEVIDENCE: first\nACCEPTANCE: x\n"
        "on reflection...\nVERDICT: stale\nCHECK: b\nEVIDENCE: second\nACCEPTANCE: -")
    assert parsed["verdict"] == "stale" and parsed["evidence"] == "second"


def test_an_invented_verdict_is_rejected():
    assert M.parse_verdict(
        "VERDICT: looks_fine\nCHECK: x\nEVIDENCE: y\nACCEPTANCE: -") is None


def test_no_block_at_all_is_none():
    assert M.parse_verdict("I had a good look and it seems fine really.") is None


@pytest.mark.parametrize("verdict", B.VERDICTS)
def test_every_declared_verdict_parses(verdict):
    parsed = M.parse_verdict(f"VERDICT: {verdict}\nCHECK: c\nEVIDENCE: e\nACCEPTANCE: -")
    assert parsed and parsed["verdict"] == verdict


# ---------------------------------------------------------------------------
# The property that matters most
# ---------------------------------------------------------------------------

def test_triage_never_starts_a_round():
    """Implementation must be a separate, explicit act.

    If triage could open a round, a wrong `confirmed` would send the gate after
    a problem that does not exist — the exact failure this pipeline is designed
    to prevent.
    """
    import inspect
    src = inspect.getsource(M)
    for forbidden in ("selfmod_start", "round.start", "R.start(", "promote("):
        assert forbidden not in src, f"triage must not call {forbidden}"


def test_the_prompt_states_that_retiring_is_a_good_outcome():
    """A pipeline that only counts code as progress turns a stale backlog into
    a pile of unnecessary changes."""
    assert "good outcome" in M.PROMPT
    assert "Never guess" in M.PROMPT
    assert "read-only" in M.PROMPT


def test_the_prompt_demands_evidence_and_an_acceptance_check():
    assert "Quote your evidence" in M.PROMPT
    assert "ACCEPTANCE:" in M.PROMPT


def test_summarize_counts_retirements_separately(backlog_dir, tmp_path):
    for i in (60, 61, 62):
        write_item(backlog_dir, i)
    ledger = tmp_path / "l.jsonl"
    ledger.write_text("\n".join(json.dumps(
        {"event": "backlog_triage", "item_id": i, "verdict": v})
        for i, v in ((60, "stale"), (61, "already_done"), (62, "confirmed"))) + "\n")
    s = B.summarize(ledger)
    assert s["retired"] == 2 and s["confirmed"] == 1


# ---------------------------------------------------------------------------
# Board scoping
#
# The backlog is shared. Of 53 open items, 3 are Alfie (robot firmware) and 1
# sits on an Architecture board — legitimately out of scope for a
# self-modification pass. The `board` field says so for free, and spending an
# LLM turn per item to rediscover it is waste: verified against #38 "Alfie —
# Fix mecanum wheels behavior", where a full triage turn correctly concluded
# `not_code` from something the frontmatter already knew.
# ---------------------------------------------------------------------------

def test_only_the_lloyd_board_is_in_scope_by_default(backlog_dir, tmp_path):
    write_item(backlog_dir, 70, board="lloyd", days_old=100)
    write_item(backlog_dir, 71, board="alfie", days_old=300)
    write_item(backlog_dir, 72, board="Architecture", days_old=250)
    assert [i.id for i in B.open_items()] == [70]


def test_an_older_out_of_scope_item_does_not_get_selected(backlog_dir, tmp_path):
    """Oldest-first must not drag in another board's work."""
    write_item(backlog_dir, 80, board="lloyd", days_old=50)
    write_item(backlog_dir, 81, board="alfie", days_old=400)
    assert B.select_candidate(tmp_path / "none.jsonl").id == 80


def test_boards_none_means_everything(backlog_dir):
    write_item(backlog_dir, 90, board="lloyd")
    write_item(backlog_dir, 91, board="alfie")
    assert len(B.open_items(None)) == 2


def test_board_matching_is_case_insensitive(backlog_dir):
    write_item(backlog_dir, 100, board="Lloyd")
    assert [i.id for i in B.open_items()] == [100]


def test_an_item_with_no_board_is_out_of_scope(backlog_dir):
    """Absent board is not the same as Lloyd's board — say so explicitly."""
    write_item(backlog_dir, 110, board="")
    assert B.open_items() == []
    assert len(B.open_items(None)) == 1


def test_the_summary_names_the_boards_it_counted(backlog_dir, tmp_path):
    write_item(backlog_dir, 120, board="lloyd")
    write_item(backlog_dir, 121, board="alfie")
    s = B.summarize(tmp_path / "none.jsonl")
    assert s["boards"] == ["lloyd"] and s["open_items"] == 1
