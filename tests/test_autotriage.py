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

from scripts.automod import backlog as B
from workers.sources import autotriage as M


def write_item(tmp_path, item_id, *, status="draft", days_old=100, body="Do the thing.",
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
    write_item(backlog_dir, 2, status="draft")
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
    assert fm["autotriage_retired"] == "stale"


def test_a_confirmed_verdict_leaves_the_item_open(backlog_dir, tmp_path):
    """Confirmed means there is real work — not that the work is finished."""
    path = write_item(backlog_dir, 51)
    item = B.load_item(path)
    B.record_verdict(item, "confirmed", "Still reproduces on HEAD.", close=True)
    fm = yaml.safe_load(path.read_text().split("---\n")[1])
    assert fm["status"] == "up_next"
    assert "autotriage_retired" not in fm


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
    assert "Automod triage" in text
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
    for forbidden in ("automod_start", "round.start", "R.start(", "promote("):
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


# ---------------------------------------------------------------------------
# Provenance: the prompt says where the item came from
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from scripts.automod import state as S  # noqa: E402
from workers.sources import _common as C  # noqa: E402


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(S, "LEDGER_PATH", path)
    return path


def _ev(ledger, **row):
    S.append_event(row, path=ledger)


def test_the_origin_block_names_who_filed_it_the_keep_prior_filings_and_findings(backlog_dir, ledger):
    write_item(backlog_dir, 7, status="done", name="Parent")
    body = "Do it.\n\n## Findings (round A)\n\n- x\n\n## Findings (triage B)\n\n- y\n"
    write_item(backlog_dir, 42, body=body, name="Child")
    _ev(ledger, event="backlog_triage", item_id=7, verdict="stale", spawned=[42], merged=[])
    _ev(ledger, event="backlog_group_triage", cluster_id="c-1", judged={"42": "keep", "43": "fold"})
    _ev(ledger, event="backlog_triage", item_id=42, verdict="incomplete", spawned=[601], merged=[])
    item = B.item_by_id(42)
    text = M.render_prompt(item, ledger=ledger)
    origin = text[text.index("<origin"):text.index("</origin>")]
    assert "filed by triage of #7 (stale)" in origin
    assert 'parent="#7 (done)"' in origin
    assert "group triage c-1 judged it `keep` on" in origin
    assert "earlier triages of this item filed or appended to: #601" in origin
    assert "2 Findings section(s)" in origin
    assert 'tags="backlog"' in origin


def test_a_human_item_says_so(backlog_dir, ledger):
    write_item(backlog_dir, 5)
    text = M.render_prompt(B.item_by_id(5), ledger=ledger)
    assert "a human's item, or a writer outside the loop" in text


def test_spawn_origin_reads_all_three_producers(ledger):
    _ev(ledger, event="backlog_implement", item_id=3, phase="finished", spawned=[30])
    _ev(ledger, event="arch_review", unit="doc:automod", verdict="stale", filed=[31])
    assert B.spawn_origin(ledger, 30)["by"] == "autocode"
    assert B.spawn_origin(ledger, 31) == {"by": "arch-review", "parent": "doc:automod",
                                          "ts": B.spawn_origin(ledger, 31)["ts"], "verdict": "stale"}
    assert B.spawn_origin(ledger, 99) is None


def test_prior_triage_spawned_keeps_ledger_order_and_skips_the_item_itself(ledger):
    _ev(ledger, event="backlog_triage", item_id=9, verdict="incomplete", spawned=[602, 9], merged=[])
    _ev(ledger, event="backlog_triage", item_id=9, verdict="stale", spawned=[601], merged=[602, 400])
    assert B.prior_triage_spawned(ledger, 9) == [602, 601, 400]


# ---------------------------------------------------------------------------
# The implement-pool depth gate
# ---------------------------------------------------------------------------

def _ready(backlog_dir, ledger, n, start=1000, **kw):
    for i in range(start, start + n):
        write_item(backlog_dir, i, status="up_next", name=f"ready {i}")
        _ev(ledger, event="backlog_triage", item_id=i, verdict="confirmed",
            acceptance=kw.get("acceptance", "it passes"))


class _QItem:
    def __init__(self, payload=None):
        self.payload = payload or {}


def _stub_turn(monkeypatch):
    async def turn(prompt, **kw):
        turn.calls.append(prompt)
        return {"text": "VERDICT: unverifiable\nSURFACE: code\nCHECK: -\nEVIDENCE: e\n"
                        "ACCEPTANCE: none\nSPAWNED: none\n",
                "session_id": "s", "stop_reason": "stop", "num_turns": 3, "errors": []}
    turn.calls = []
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    return turn


def test_a_full_pool_pauses_single_triage_and_says_both_numbers(backlog_dir, ledger, monkeypatch):
    _ready(backlog_dir, ledger, 40)
    write_item(backlog_dir, 7, days_old=300)
    turn = _stub_turn(monkeypatch)
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["status"] == "skipped"
    assert "single-item triage paused: 40 ready in up_next ≥ bound 40" in out["summary"]
    assert "0 items landed in 7 d, floor 40" in out["summary"]
    assert turn.calls == []
    assert [e for e in S.read_events(path=ledger) if e.get("item_id") == 7] == [], \
        "a skipped run is not a triage"
    assert B.select_candidate(ledger).id == 7, "the draft is still a candidate"


def test_items_autocode_would_not_take_do_not_fill_the_pool(backlog_dir, ledger, monkeypatch):
    _ready(backlog_dir, ledger, 39)
    _ready(backlog_dir, ledger, 3, start=2000, acceptance="human-only: config.yaml")
    for i in (3000, 3001):     # grouped members parked in up_next
        p = write_item(backlog_dir, i, status="up_next")
        B.update_frontmatter(p, {"group": 1})
        _ev(ledger, event="backlog_triage", item_id=i, verdict="confirmed", acceptance="x")
    write_item(backlog_dir, 7, days_old=300)
    turn = _stub_turn(monkeypatch)
    out = asyncio.run(M.execute(_QItem({"group_triage": False, "implement_pool_floor": 40})))
    assert out["status"] == "success" and out["item_id"] == 7
    assert len(turn.calls) == 1


def test_the_bound_is_the_floor_until_landings_exceed_it(ledger):
    now = datetime.now(timezone.utc).timestamp()
    for i, item in enumerate((1, 2, 3)):
        _ev(ledger, event="vault_land", ok=True, item_id=item, commit=f"c{i}")
    assert B.implement_pool_bound(ledger, floor=20, now=now + 1) == {
        "bound": 20, "floor": 20, "landed_items_7d": 3}
    assert B.implement_pool_bound(ledger, floor=2, now=now + 1)["bound"] == 3


def test_landed_items_are_distinct_items_not_rounds(ledger):
    """#487 landed several times in three days; a pool sized by rows is three
    times too deep."""
    for commit in ("a", "b", "c"):
        _ev(ledger, event="vault_land", ok=True, item_id=487, commit=commit)
    _ev(ledger, event="vault_land", ok=False, item_id=488, commit="d")
    _ev(ledger, event="backlog_implement", item_id=9, phase="finished", round_id="SM_1")
    _ev(ledger, event="promoted", round_id="SM_1", commit="sha1")
    _ev(ledger, event="settled", commit="sha1")
    assert B.landed_items_trailing(ledger, 7) == 2
    later = datetime.now(timezone.utc).timestamp() + 8 * 86400
    assert B.landed_items_trailing(ledger, 7, now=later) == 0


def test_group_triage_still_runs_under_a_full_pool(backlog_dir, ledger, monkeypatch):
    from scripts.automod import cluster as CL
    _ready(backlog_dir, ledger, 40)
    monkeypatch.setattr(B, "select_cluster", lambda *a, **k: ({"id": "c-9"}, ["m"]))
    monkeypatch.setattr(CL, "load_clusters", lambda *a, **k: {"clusters": []})

    async def group(item, cluster, members):
        return {"status": "success", "cluster_id": cluster["id"]}
    monkeypatch.setattr(M, "_execute_group", group)
    out = asyncio.run(M.execute(_QItem({"group_triage": True, "implement_pool_floor": 20})))
    assert out == {"status": "success", "cluster_id": "c-9"}


def test_the_floor_rides_in_the_payload():
    import inspect
    src = inspect.getsource(M.enqueue_if_due)
    assert '"implement_pool_floor"' in src and '"spawn_cap"' in src
