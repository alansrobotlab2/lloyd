"""Verdict → outcome join, the worth hit rate, scorecard row 23, and the sweep's
"a `swept` tag is a rank only when there is a rank" fix.

Everything runs on rows and item files written here: no live ledger, no clock.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, scorecard as SC, state as S
from scripts.automod import verdict_outcomes as VO

NOW = 1_800_000_000.0
DAY = 86400.0


def _item(d: Path, item_id: int, *, status="done", tags=("backlog",), body="Body.", **fm_extra):
    fm = {"status": status, "priority": "low", "board": "lloyd", "tags": list(tags),
          "created": "2026-01-01T00:00:00", **fm_extra}
    (d / f"{item_id}-x.md").write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Item {item_id}\n\n{body}\n",
        encoding="utf-8")


def _row(days_ago: float, event: str, **fields) -> dict:
    return {"ts": NOW - days_ago * DAY, "event": event, **fields}


@pytest.fixture
def board(tmp_path):
    d = tmp_path / "backlog"
    d.mkdir()
    return d


def _judged(rows, board, **kw):
    return {(v["decider"], v["verdict"], v["item_id"]): v
            for v in VO.judged(days=30, rows=rows, backlog_dir=board, now=NOW, **kw)}


# ── the vocabulary is read off the rows ─────────────────────────────────────

def test_each_decider_is_read_off_its_own_row_shape():
    rows = [
        _row(5, "backlog_triage", item_id=1, verdict="confirmed", closed=False),
        _row(5, "backlog_triage", item_id=2, verdict="stale", closed=True, sweep_batch="sw-1"),
        _row(5, "backlog_triage", item_id=3, verdict="folded", closed=False, group=9),
        _row(5, "backlog_triage", item_id=4, verdict="confirmed", closed=False, red_tree=True),
        _row(5, "backlog_triage", item_id=5, verdict="unverifiable"),
        _row(5, "backlog_triage", item_id=6, verdict="incomplete"),
        _row(5, "backlog_sweep", batch_id="sw-1", judged={"7": "keep", "8": "keep", "2": "stale"},
             ranked={"7": "high/small", "8": "low/medium"}),
        _row(5, "owed_check", item_id=9, ok=True, result={"moved": "reopened"}),
        _row(5, "owed_check", item_id=10, ok=True, result={"moved": "closed"}),
        _row(5, "owed_check", item_id=11, ok=True, result={"moved": ""}),
        _row(5, "steward_move", item_id=12, to="up_next", **{"from": "draft"}),
        _row(50, "backlog_triage", item_id=13, verdict="confirmed"),        # out of window
    ]
    got = {(v["decider"], v["verdict"], v["cls"], v["item_id"])
           for v in VO.verdicts_in(rows, since=NOW - 30 * DAY, until=NOW)}
    assert got == {
        ("triage", "confirmed", VO.CONFIRM, 1), ("sweep", "stale", VO.RETIRE, 2),
        ("group_triage", "folded", VO.FOLD, 3), ("gate", "confirmed", VO.CONFIRM, 4),
        ("triage", "unverifiable", VO.OTHER, 5),
        ("sweep", "keep", VO.CONFIRM, 7), ("sweep", "keep_low", VO.CONFIRM, 8),
        ("owed_check", "reopen", VO.CONFIRM, 9), ("owed_check", "close", VO.RETIRE, 10),
        ("board_steward", "to_up_next", VO.CONFIRM, 12)}


# ── outcomes ────────────────────────────────────────────────────────────────

def test_a_confirmation_is_joined_to_what_the_ledger_says_happened_next(board):
    for i in (1, 2, 3, 4, 5, 6):
        _item(board, i)
    _item(board, 7, status="draft")
    _item(board, 8, status="up_next")
    confirm = lambda i: _row(20, "backlog_triage", item_id=i, verdict="confirmed", closed=False)  # noqa: E731
    rows = [confirm(i) for i in range(1, 9)] + [
        # 1 landed and stayed
        _row(19, "backlog_implement", item_id=1, phase="started"),
        _row(19, "review", item_id=1, blocking=True),
        _row(18.5, "backlog_implement", item_id=1, phase="started"),
        _row(18, "item_landed", item_id=1, closed=True, commit="aaa"),
        # 2 landed, then a round started on it again
        _row(18, "item_landed", item_id=2, closed=True, commit="bbb"),
        _row(10, "backlog_implement", item_id=2, phase="started"),
        # 3 landed, then the guardian took the commit off main
        _row(18, "item_landed", item_id=3, closed=True, commit="ccc"),
        _row(17.9, "rollback_succeeded", commit="ccc"),
        # 4 the round found it unnecessary
        _row(18, "item_closed", item_id=4, by="autocode", acceptance="unnecessary"),
        # 5 a later triage retired it
        _row(18, "backlog_triage", item_id=5, verdict="stale", closed=True),
        # 6 owed-check closed it with no landing
        _row(18, "owed_check", item_id=6, ok=True, result={"moved": "closed", "remaining": 0}),
        # 7 its attempts are spent and it waits in draft
        _row(18, "status_moved", item_id=7, to="draft",
             reason="its unattended attempts are spent; owed-check decides"),
    ]
    got = _judged(rows, board)
    out = {i: got[("triage", "confirmed", i)]["outcome"] for i in range(1, 9)}
    assert out == {1: "landed_kept", 2: "landed_reopened", 3: "landed_reopened",
                   4: "round_rejected", 5: "retired_later", 6: "owed_closed",
                   7: "spent", 8: "open"}
    one = got[("triage", "confirmed", 1)]
    assert (one["attempts"], one["refusals"]) == (2, 1)
    assert one["closed_ts"] == NOW - 18 * DAY
    assert got[("triage", "confirmed", 3)]["detail"] == "guardian rollback"
    assert got[("triage", "confirmed", 4)]["detail"] == "unnecessary"


def test_a_landing_before_the_verdict_is_not_that_verdicts_outcome(board):
    _item(board, 1, status="up_next")
    rows = [_row(25, "item_landed", item_id=1, closed=True, commit="aaa"),
            _row(20, "backlog_triage", item_id=1, verdict="confirmed", closed=False)]
    assert _judged(rows, board)[("triage", "confirmed", 1)]["outcome"] == "open"


def test_a_retirement_that_came_back_is_reopened_and_one_that_did_not_stayed_closed(board):
    _item(board, 1)
    _item(board, 2)
    _item(board, 3, status="draft")           # reopened by hand: no ledger row at all
    retire = lambda i: _row(20, "backlog_triage", item_id=i, verdict="stale", closed=True)  # noqa: E731
    rows = [retire(1), retire(2), retire(3),
            _row(15, "backlog_triage", item_id=2, verdict="confirmed", closed=False),
            _row(14, "item_landed", item_id=2, closed=True, commit="abc")]
    got = _judged(rows, board)
    assert got[("triage", "stale", 1)]["outcome"] == "stayed_closed"
    assert got[("triage", "stale", 2)]["outcome"] == "reopened"
    assert got[("triage", "stale", 2)]["detail"] == "landed afterwards"
    assert got[("triage", "stale", 3)]["detail"] == "open on the board"


def test_a_folded_member_lands_when_its_umbrella_closes_it(board):
    _item(board, 1)
    rows = [_row(20, "backlog_triage", item_id=1, verdict="folded", closed=False, group=9),
            _row(18, "item_closed", item_id=1, by="umbrella", umbrella_id=9)]
    assert _judged(rows, board)[("group_triage", "folded", 1)]["outcome"] == "landed_kept"


def test_one_item_counts_once_per_decider_and_verdict_and_a_duplicate_row_is_one_event(board, tmp_path):
    _item(board, 1)
    rows = [_row(20, "backlog_triage", item_id=1, verdict="confirmed", closed=False),
            _row(15, "backlog_triage", item_id=1, verdict="confirmed", closed=False),
            _row(10, "item_landed", item_id=1, closed=True, commit="aaa")]
    reading = VO.compute(days=30, rows=rows, backlog_dir=board, now=NOW)
    assert reading["verdicts"] == 1
    assert reading["by_decider"]["triage/confirmed"]["landed_kept"] == 1
    # the live file and a rotation both carrying a row: read once
    ledger = tmp_path / "promotions.jsonl"
    for r in rows + rows[:1]:
        S.append_event(dict(r), path=ledger)
    assert len(VO.ledger_rows(ledger)) == len(rows)


# ── rates ───────────────────────────────────────────────────────────────────

def test_a_rate_needs_min_n_and_open_items_are_not_in_its_denominator(board):
    rows = []
    for i in range(1, VO.MIN_N + 1):
        _item(board, i)
        rows.append(_row(20, "backlog_triage", item_id=i, verdict="confirmed", closed=False))
        if i > 5:
            rows.append(_row(18, "item_landed", item_id=i, closed=True, commit=f"c{i}"))
        else:
            rows.append(_row(18, "item_closed", item_id=i, by="autocode", acceptance="rejected"))
    for i in range(100, 110):                       # ten still open: censored, not failures
        _item(board, i, status="up_next")
        rows.append(_row(20, "backlog_triage", item_id=i, verdict="confirmed", closed=False))
    _item(board, 200)
    rows.append(_row(20, "backlog_triage", item_id=200, verdict="stale", closed=True))
    reading = VO.compute(days=30, rows=rows, backlog_dir=board, now=NOW)
    cell = reading["deciders"]["triage"][VO.CONFIRM]
    assert (cell["n"], cell["resolved"], cell["open"]) == (VO.MIN_N + 10, VO.MIN_N, 10)
    assert cell["landed_kept_rate"]["rate"] == round((VO.MIN_N - 5) / VO.MIN_N, 4)
    lo, hi = cell["landed_kept_rate"]["ci"]
    assert lo < cell["landed_kept_rate"]["rate"] < hi
    retire = reading["deciders"]["triage"][VO.RETIRE]["reopened_rate"]
    assert (retire["k"], retire["n"], retire["rate"]) == (0, 1, None)     # one row is not a rate
    summary = VO.scorecard_summary(reading)["triage"]
    assert summary["confirm_landed_kept"]["rate"] is not None
    assert summary["retire_reopened"]["rate"] is None


def test_wilson_matches_the_reference_values_and_separation_needs_disjoint_intervals():
    assert VO.wilson(10, 20) == (0.2993, 0.7007)          # eval/stats.py's own worked example
    assert VO.wilson(20, 20)[0] == 0.8389
    assert VO.wilson(0, 0) is None
    a, b = VO._rate(90, 100), VO._rate(50, 100)
    assert VO.separated(a, b) and not VO.separated(a, VO._rate(88, 100))
    assert not VO.separated(a, VO._rate(1, 2))            # under min n: no claim either way


def test_outcomes_are_grouped_by_who_filed_the_item_once_per_item(board):
    _item(board, 1, tags=("backlog", B.OWED_SPAWN_TAG))
    _item(board, 2, tags=("backlog", "spawned-by-triage"))
    rows = [_row(20, "backlog_triage", item_id=1, verdict="confirmed", closed=False),
            _row(19, "backlog_sweep", batch_id="sw-1", judged={"1": "keep"}, ranked={"1": "high/small"}),
            _row(18, "item_landed", item_id=1, closed=True, commit="aaa"),
            _row(20, "backlog_triage", item_id=2, verdict="stale", closed=True)]
    reading = VO.compute(days=30, rows=rows, backlog_dir=board, now=NOW)
    assert reading["by_source"]["owed-check/confirm"]["n"] == 1        # two deciders, one item
    assert reading["by_source"]["triage/retire"]["stayed_closed"] == 1
    assert "landed-kept of resolved" in VO.render(reading)


# ── worth ───────────────────────────────────────────────────────────────────

def test_the_worth_table_separates_the_order_a_label_caused_from_what_it_predicts(board):
    rows = []
    n = 0

    def keep(worth, *, attempted, landed):
        nonlocal n
        n += 1
        _item(board, n, status="done" if landed else "up_next")
        rows.append(_row(20, "backlog_sweep", batch_id=f"sw-{n}", judged={str(n): "keep"},
                         ranked={str(n): f"{worth}/small"}))
        if attempted:
            rows.append(_row(19, "backlog_implement", item_id=n, phase="started"))
        if landed:
            rows.append(_row(18, "item_landed", item_id=n, closed=True, commit=f"c{n}"))

    for _ in range(30):
        keep("high", attempted=True, landed=True)
    for _ in range(20):
        keep("low", attempted=True, landed=True)
    for _ in range(40):
        keep("low", attempted=False, landed=False)       # never reached: the order, not the item
    reading = VO.worth_hit_rate(days=30, rows=rows, backlog_dir=board, now=NOW)
    high, low = reading["worth"]["high"], reading["worth"]["low"]
    assert (high["ranked"], high["attempted"]) == (30, 30)
    assert (low["ranked"], low["attempted"]) == (60, 20)
    assert low["landed_kept_of_ranked"]["rate"] == round(20 / 60, 4)
    assert low["landed_kept_of_attempted"]["rate"] == 1.0
    assert high["attempts_per_landing"] == 1.0 and high["median_days_to_land"] == 2.0
    assert reading["worth_separated"]["landed_kept_of_attempted"]["high_vs_low"] is False
    assert reading["size"]["small"]["ranked"] == 90
    assert "Worth pairs" in VO.render_worth(reading)
    # a keep too young to have an outcome can be left out
    assert VO.worth_hit_rate(days=30, min_age_days=25, rows=rows, backlog_dir=board,
                             now=NOW)["ranked"] == 0


def test_rank_key_carries_no_worth_correction():
    """Measured 2026-11-22 over 2,521 sweep keeps at least 14 d old: among
    attempted items `high`, `medium` and `low` land and stay landed 92 %, 93 %
    and 93 % of the time. No label is inverted, so no correction was built —
    and none may appear without the measurement that argues for it."""
    import inspect
    assert "worth_correction" not in inspect.getsource(B.rank_key)
    assert B._WORTH_ORDER == {"high": 0, "medium": 1, "": 2, "low": 3}


# ── scorecard row 23 ────────────────────────────────────────────────────────

def test_row_23_says_unmeasured_under_the_minimum_and_a_rate_above_it(tmp_path):
    row = {"verdict_outcomes": {"days": 30.0, "verdicts": 130, "min_n": 20, "deciders": {
        "triage": {"confirm_landed_kept": {"k": 86, "n": 100, "rate": 0.86, "ci": (0.78, 0.92)},
                   "retire_reopened": {"k": 0, "n": 25, "rate": 0.0, "ci": (0.0, 0.13)}},
        "board_steward": {"confirm_landed_kept": {"k": 3, "n": 5, "rate": None, "ci": None},
                          "retire_reopened": None}}}}
    line = SC._render_verdict_outcomes(row)
    assert line.startswith("| 23 | verdict outcomes | 130 verdicts / 30 d |")
    assert "triage: confirmed→landed-kept 86% (86/100), retired→reopened 0% (0/25)" in line
    assert "board_steward: confirmed→landed-kept unmeasured (3/5), retired→reopened none" in line
    assert SC._render_verdict_outcomes({}) == "| 23 | verdict outcomes | — | not measured for this row |"
    # appended after rows 18-21, on a row computed over nothing at all
    empty = tmp_path / "ledger.jsonl"
    empty.write_text("")
    full = SC.compute(since_days=7, ledger=empty, backlog_dir=tmp_path / "nope",
                      repo=tmp_path / "norepo", now=NOW)
    assert full["verdict_outcomes"] == {}
    assert "| 23 | verdict outcomes | — | not measured for this row |" in SC.render(full).splitlines()
    assert line in SC.render({**full, **row}).splitlines()


def test_row_23_is_computed_from_the_ledger_and_reused_within_its_ttl(board, tmp_path, monkeypatch):
    ledger = tmp_path / "promotions.jsonl"
    _item(board, 1)
    S.append_event({"event": "backlog_triage", "item_id": 1, "verdict": "confirmed",
                    "closed": False}, path=ledger)
    S.append_event({"event": "item_landed", "item_id": 1, "closed": True, "commit": "aaa"},
                   path=ledger)
    monkeypatch.setattr(SC, "_VO_MEMO", {})
    import time
    now = time.time() + 5
    out = SC._verdict_outcomes(ledger, board, now)
    cell = out["deciders"]["triage"]["confirm_landed_kept"]
    assert (cell["k"], cell["n"], cell["rate"]) == (1, 1, None)
    calls = []
    monkeypatch.setattr(VO, "compute", lambda **kw: calls.append(kw) or {})
    assert SC._verdict_outcomes(ledger, board, now + 60) is out          # inside the TTL
    assert calls == []
    SC._verdict_outcomes(ledger, board, now + SC.VO_TTL_S + 1)
    assert len(calls) == 1
    assert SC._verdict_outcomes(ledger, tmp_path / "missing", now) == {}


# ── the sweep: a `swept` tag with no rank is not a rank ─────────────────────

@pytest.fixture
def sweep_board(board, tmp_path, monkeypatch):
    monkeypatch.setattr(B, "BACKLOG_DIR", board)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(B, "sweep_enabled", lambda: True)
    return board


def test_an_item_filed_with_a_copied_swept_tag_is_still_read_by_the_sweep(sweep_board):
    """#4225 on 2026-11-22: a triage follow-up filed with its parent's tags —
    quarantined, never swept, and exempt from expiry on the tag. In no pool."""
    _item(sweep_board, 1, status="draft", tags=("spawned-by-triage", B.SWEPT_TAG))
    _item(sweep_board, 2, status="draft", tags=("backlog", B.SWEPT_TAG), worth="medium", size="small")
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [1]
    assert B.has_standing_rank(B.item_by_id(2)) and not B.has_standing_rank(B.item_by_id(1))
    # once a sweep has really ruled on it, the ledger row settles it
    S.append_event({"event": "backlog_sweep", "batch_id": "sw-1", "item_ids": [1],
                    "judged": {"1": "keep"}, "ranked": {"1": "medium/small"}}, path=S.LEDGER_PATH)
    assert B.sweep_pool(S.LEDGER_PATH) == []


def test_a_keep_with_no_worth_is_not_recorded_and_the_item_is_offered_again(sweep_board):
    _item(sweep_board, 1, status="draft")
    _item(sweep_board, 2, status="draft")
    _item(sweep_board, 3, status="draft")
    members = [B.item_by_id(i) for i in (1, 2, 3)]
    out = B.record_sweep_verdicts("sw-1", members, {
        1: {"verdict": "keep", "worth": "", "size": "small", "evidence": "no rank written"},
        2: {"verdict": "keep", "worth": "high", "size": "small", "evidence": "a real bug"},
        3: {"verdict": "stale", "worth": "", "size": "", "evidence": "premise gone"}},
        ledger=S.LEDGER_PATH)
    assert out["judged"] == {"2": "keep", "3": "stale"}
    assert out["ranked"] == {"2": "high/small"} and out["kept"] == 1
    assert B.SWEPT_TAG not in B.item_by_id(1).tags
    assert B.SWEPT_TAG in B.item_by_id(2).tags
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [1]
