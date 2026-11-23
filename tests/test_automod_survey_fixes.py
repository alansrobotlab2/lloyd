"""The 2026-11-22 survey of the automod loop: what was real, pinned.

A code survey listed ten suspected defects. Each was checked against the code
and against the whole promotions ledger (live file plus three rotations, 91,527
rows, 37,082 of them `gate`). Most were not what the survey said; the ones with
a real residue are fixed and pinned here. `architecture/automod.md` §3.2l has
the measurements for all ten, including the ones left alone.
"""
from __future__ import annotations

import ast
import gzip
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as MB
from scripts.automod import backlog as B, promote as P, round as R, state as S, worktree as W

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "CURRENT_PATH", tmp_path / "current.json")
    S.ledger_rows_reset()
    return d


def _ev(**row):
    S.append_event(row, path=S.LEDGER_PATH)


def _rows(event: str) -> list[dict]:
    return [r for r in S.ledger_rows(S.LEDGER_PATH) if r.get("event") == event]


def write_item(d: Path, item_id, *, status="draft", tags=("backlog",)) -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    fm = {"status": status, "priority": "medium", "created": created,
          "board": "lloyd", "tags": list(tags)}
    p = d / f"{item_id}-item-{item_id}.md"
    p.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Item {item_id}\n\nDo it.\n",
                 encoding="utf-8")
    return p


# ── 1. a landing that stops before the merge leaves a row ───────────────────

def _passed_gate(rid: str) -> None:
    (S.ROUNDS_DIR / rid).mkdir(parents=True)
    (S.ROUNDS_DIR / rid / "gate.json").write_text(json.dumps(
        {"ok": True, "base": "b" * 40, "head": "h" * 40, "changed_paths": ["tests/x.py"],
         "rungs": []}))


@pytest.fixture
def landing(monkeypatch):
    """`R.land` with everything around the promoter stubbed: the promoter's
    own raise is the thing under test."""
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    monkeypatch.setattr(P, "restart_needed", lambda changed: (False, "inert"))
    monkeypatch.setattr(P, "landing_is_eager", lambda changed, report: [])

    class _Lock:
        def release(self):
            pass
    monkeypatch.setattr(R, "_land_lock", lambda *a, **k: _Lock())
    monkeypatch.setattr(W, "remove", lambda *a, **k: None)


def test_a_bare_refusal_before_the_merge_is_recorded_and_changes_no_verdict(landing, monkeypatch):
    _passed_gate("SM_HALT")

    def halted(*a, **k):
        raise P.PromoteError("promotions are halted: /state/promotions-halted")
    monkeypatch.setattr(P, "promote", halted)
    with pytest.raises(P.PromoteError, match="halted"):
        R.land("SM_HALT")
    (row,) = _rows(R.LAND_REFUSED_EVENT)
    assert row["round_id"] == "SM_HALT" and row["error_type"] == "PromoteError"
    assert "halted" in row["detail"] and row["ok"] is False
    # Not a `land_failed`: the reaper's one-rescue rule and the attempt
    # accounting both key on that event, and neither may move.
    assert _rows("land_failed") == []
    assert "SM_HALT" not in B._last_gate_per_round(S.LEDGER_PATH)
    assert S.read_land_marker("SM_HALT") is None, "the marker is still cleared"


def test_an_unanticipated_exception_is_recorded_with_its_type(landing, monkeypatch):
    _passed_gate("SM_BOOM")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(["git", "rev-parse", "HEAD"], W.GIT_READ_TIMEOUT_S)
    monkeypatch.setattr(P, "promote", boom)
    with pytest.raises(subprocess.TimeoutExpired):
        R.land("SM_BOOM")
    (row,) = _rows(R.LAND_REFUSED_EVENT)
    assert row["error_type"] == "TimeoutExpired" and "timed out" in row["detail"]


def test_a_refusal_the_promoter_already_recorded_is_not_recorded_twice(landing, monkeypatch):
    _passed_gate("SM_REC")

    def refused(round_id, *a, **k):
        P._land_failed(round_id, "abc12345 is on the rollback denylist", external=False)
    monkeypatch.setattr(P, "promote", refused)
    with pytest.raises(P.PromoteError):
        R.land("SM_REC")
    assert len(_rows("land_failed")) == 1
    assert _rows(R.LAND_REFUSED_EVENT) == []


def test_nothing_is_recorded_after_the_merge_or_on_a_dry_run(landing, monkeypatch):
    _passed_gate("SM_POST")

    def after_merge(*a, **k):
        P.mark_merged(True)
        raise P.PromoteError("backend never became healthy after restart")
    monkeypatch.setattr(P, "promote", after_merge)
    with pytest.raises(P.PromoteError):
        R.land("SM_POST")
    assert _rows(R.LAND_REFUSED_EVENT) == [], "post-merge failures are the promoter's to record"

    _passed_gate("SM_DRY")

    def halted(*a, **k):
        raise P.PromoteError("promotions are halted")
    monkeypatch.setattr(P, "promote", halted)
    with pytest.raises(P.PromoteError):
        R.land("SM_DRY", dry_run=True)
    assert _rows(R.LAND_REFUSED_EVENT) == []


def test_a_missing_gate_report_is_recorded_too(landing):
    with pytest.raises(RuntimeError, match="no gate report"):
        R.land("SM_NOGATE")
    (row,) = _rows(R.LAND_REFUSED_EVENT)
    assert row["error_type"] == "RuntimeError"


def test_a_record_that_cannot_be_written_does_not_mask_the_refusal(landing, monkeypatch, capsys):
    _passed_gate("SM_DISK")

    def halted(*a, **k):
        raise P.PromoteError("promotions are halted")
    monkeypatch.setattr(P, "promote", halted)
    real = S.append_event

    def full_disk(entry, path=None):
        if entry.get("event") == R.LAND_REFUSED_EVENT:
            raise OSError(28, "No space left on device")
        return real(entry, path)
    monkeypatch.setattr(S, "append_event", full_disk)
    with pytest.raises(P.PromoteError, match="halted"):
        R.land("SM_DISK")
    assert "could not record the refusal of SM_DISK" in capsys.readouterr().err


# ── 3. the sweep does not retire an item whose round is running ─────────────

def _keep(n):
    return {"verdict": "keep", "worth": "high", "size": "small", "evidence": f"k{n}"}


def _stale(n):
    return {"verdict": "stale", "evidence": f"gone {n}"}


def test_the_sweep_pool_skips_an_item_an_implement_turn_holds(isolated):
    write_item(isolated, 1, status="up_next")
    write_item(isolated, 2, status="up_next")
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [1, 2]
    assert B.claim_for_implement("autocode:round:1", 2) is None
    try:
        assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [1]
    finally:
        B.release_implement_claim("autocode:round:1")
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [1, 2]


def test_a_retirement_of_an_item_with_a_turn_in_flight_is_dropped_not_applied(isolated, caplog):
    write_item(isolated, 1, status="up_next")
    write_item(isolated, 2, status="up_next")
    write_item(isolated, 3)
    _ev(event="backlog_implement", item_id=1, phase="started", session_id="s1")
    members = [B.item_by_id(i) for i in (1, 2, 3)]
    with caplog.at_level(logging.WARNING, logger=B.logger.name):
        res = B.record_sweep_verdicts("sw-t", members, {1: _stale(1), 2: _stale(2), 3: _keep(3)})
    assert B.item_by_id(1).status == "up_next", "mid-round: not closed"
    assert B.item_by_id(2).status == "done", "no round: retired as before"
    assert res["retired"] == 1 and res["kept"] == 1
    assert sorted(res["judged"]) == ["2", "3"]
    assert 1 not in B.swept_ids(S.LEDGER_PATH), "unjudged, so it is offered again"
    assert "#1 was judged stale while its round is still running" in caplog.text


def test_a_claim_with_no_ledger_row_yet_holds_a_retirement_and_a_keep_still_ranks(isolated):
    write_item(isolated, 1, status="up_next")
    write_item(isolated, 2, status="up_next")
    B.claim_for_implement("autocode:round:0", 1)
    B.claim_for_implement("autocode:round:1", 2)
    try:
        members = [B.item_by_id(i) for i in (1, 2)]
        res = B.record_sweep_verdicts("sw-c", members, {1: {"verdict": "duplicate_of",
                                                            "duplicate_of": 2, "evidence": "same"},
                                                        2: _keep(2)})
    finally:
        B.release_implement_claim("autocode:round:0")
        B.release_implement_claim("autocode:round:1")
    assert B.item_by_id(1).status == "up_next" and res["duplicates"] == 0
    assert res["kept"] == 1 and B.SWEPT_TAG in B.item_by_id(2).tags, "a ranking is safe beside a round"


def test_a_finished_round_no_longer_holds_the_item(isolated):
    write_item(isolated, 1, status="up_next")
    _ev(event="backlog_implement", item_id=1, phase="started", session_id="s1")
    _ev(event="backlog_implement", item_id=1, phase="finished", session_id="s1",
        stop_reason="stop", outcome={"landed": False, "acceptance": "not_met"})
    res = B.record_sweep_verdicts("sw-f", [B.item_by_id(1)], {1: _stale(1)})
    assert res["retired"] == 1 and B.item_by_id(1).status == "done"


def test_an_unreadable_mid_round_set_retires_nothing(isolated, monkeypatch):
    write_item(isolated, 1, status="up_next")
    write_item(isolated, 2)

    def broken(*a, **k):
        raise OSError("ledger unreadable")
    monkeypatch.setattr(B, "implement_history", broken)
    out = B.sweep_verdicts_not_mid_round({1: _stale(1), 2: _keep(2)}, {1, 2}, S.LEDGER_PATH)
    assert sorted(out) == [2], "unsure means unfinished"


def test_a_batch_with_no_retirement_never_asks_who_is_mid_round(isolated, monkeypatch):
    def never(*a, **k):
        raise AssertionError("a keep-only batch must not pay for the read")
    monkeypatch.setattr(B, "implement_history", never)
    out = B.sweep_verdicts_not_mid_round({1: _keep(1), 9: _stale(9)}, {1}, S.LEDGER_PATH)
    assert sorted(out) == [1], "and a verdict for a non-member is still dropped"


# ── 5. two caps that the 14-day rotation used to reset ──────────────────────

def _archive(tmp_path: Path, rows: list[dict]) -> None:
    lines = "".join(json.dumps({"ts": 1.0, "created_at": "2026-09-01T00:00:00Z", **r}) + "\n"
                    for r in rows)
    (tmp_path / "promotions-archive-202609.jsonl.gz").write_bytes(gzip.compress(lines.encode()))
    S.ledger_rows_reset()


def test_a_retriage_rolled_into_an_archive_still_counts_against_the_cap(tmp_path):
    _ev(event="backlog_retriage", item_id=8)
    _archive(tmp_path, [{"event": "backlog_retriage", "item_id": 7, "round_id": "SM_OLD"}])
    counts = B.retriage_counts(S.LEDGER_PATH)
    assert counts == {7: 1, 8: 1}
    assert counts[7] >= B.RETRIAGE_CAP, "the second spend is owed-check's, not a second re-triage"


def test_a_sweep_batch_abandoned_before_the_rotation_stays_out_of_the_pool(isolated, tmp_path):
    write_item(isolated, 5)
    write_item(isolated, 6)
    _archive(tmp_path, [{"event": "backlog_sweep", "batch_id": "sw-old", "item_ids": [5],
                         "verdict": "abandoned", "judged": {}}])
    assert B.sweep_abandoned_ids(S.LEDGER_PATH) == {5}
    assert [i.id for i in B.sweep_pool(S.LEDGER_PATH)] == [6]


# ── 6. read-only git calls under the landing lock are bounded ───────────────

def _raw_git_calls(path: Path) -> list[tuple[int, list[str], bool]]:
    """`(line, argv words, has timeout)` for every `subprocess.run(["git", ...])`."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "subprocess.run"):
            continue
        argv = node.args[0] if node.args else None
        if not (isinstance(argv, ast.List) and argv.elts
                and isinstance(argv.elts[0], ast.Constant) and argv.elts[0].value == "git"):
            continue
        words = [e.value for e in argv.elts if isinstance(e, ast.Constant)]
        out.append((node.lineno, words, any(k.arg == "timeout" for k in node.keywords)))
    return out


def test_the_promoters_pre_merge_head_reads_carry_the_shared_timeout():
    src = (ROOT / "scripts/automod/promote.py").read_text(encoding="utf-8")
    assert src.count("timeout=W.GIT_READ_TIMEOUT_S") >= 2
    assert 60.0 <= W.GIT_READ_TIMEOUT_S <= 600.0, "a hang detector, not a budget"
    assert W.GIT_READ_TIMEOUT_S == 120.0 == W.git.__kwdefaults__["timeout"], "one bound for both spellings"


def test_no_merge_is_ever_killed_by_a_timeout():
    """`subprocess.run(timeout=)` kills the child. A fast-forward killed part-way
    leaves production's tree half moved under an unmoved HEAD."""
    merges = [(line, has) for path in ("promote.py", "round.py")
              for line, words, has in _raw_git_calls(ROOT / "scripts/automod" / path)
              if "merge" in words]
    assert len(merges) == 2, merges
    assert not any(has for _line, has in merges)


def test_the_round_clis_lock_holding_git_reads_are_bounded():
    calls = _raw_git_calls(ROOT / "scripts/automod/round.py")
    unbounded = [(line, words) for line, words, has in calls if not has]
    # The one left unbounded is `bless`'s diff, which fails CLOSED on an
    # unreadable diff by returncode and must not start raising instead.
    assert [w for _l, w in unbounded] == [["git", "-C", "diff", "--name-only"]], unbounded


# ── 7. no whole-ledger or whole-board read on the backend's event loop ──────

#: Readers that decode the whole ledger (0.4 s after any append, measured on the
#: 21.6 MB live file) or glob the whole board. `record_sweep_verdicts` is here
#: because its mid-round guard reads the merged implement history.
_BLOCKING_READERS = {"incomplete_counts", "no_measurement_attempts", "_ledger_events",
                     "group_triage_events", "open_items", "record_sweep_verdicts",
                     "implement_history", "implement_outcomes", "desired_statuses"}


def test_the_triage_source_makes_none_of_those_reads_directly_in_a_coroutine():
    tree = ast.parse((ROOT / "workers/sources/autotriage.py").read_text(encoding="utf-8"))
    direct: list[tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[bool] = []

        def visit_AsyncFunctionDef(self, node):
            self.stack.append(True)
            self.generic_visit(node)
            self.stack.pop()

        def _sync(self, node):
            self.stack.append(False)
            self.generic_visit(node)
            self.stack.pop()
        visit_FunctionDef = visit_Lambda = _sync

        def visit_Call(self, node):
            f = node.func
            if (self.stack and self.stack[-1] and isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Name) and f.value.id == "B"
                    and f.attr in _BLOCKING_READERS):
                direct.append((node.lineno, f.attr))
            self.generic_visit(node)
    V().visit(tree)
    assert direct == [], f"called on the event loop, not through asyncio.to_thread: {direct}"


def test_group_triage_events_is_the_same_read_the_inline_filter_made():
    _ev(event="backlog_group_triage", cluster_id="c-1", verdict=B.INCOMPLETE, judged={})
    _ev(event="backlog_triage", item_id=1, verdict="confirmed")
    assert [d["cluster_id"] for d in B.group_triage_events(S.LEDGER_PATH)] == ["c-1"]


# ── 8. a fail-open step that failed says so ─────────────────────────────────

def test_a_failed_dedupe_still_lets_the_write_through_and_is_logged_and_journaled(
        monkeypatch, caplog, tmp_path):
    log = tmp_path / "dedupe.jsonl"
    monkeypatch.setattr(MB.SIM, "DEDUPE_LOG", log)

    def broken():
        raise KeyError("backlog_dedupe")
    monkeypatch.setattr(MB.SIM, "dedupe_config", broken)
    with caplog.at_level(logging.WARNING, logger=MB.logger.name):
        assert MB._dedupe({"name": "x", "description": "y"}) == ([], None, None)
    assert "dedupe failed and was skipped (KeyError" in caplog.text
    (row,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert row["action"] == "dedupe_failed" and row["error_type"] == "KeyError"


def test_a_failed_promotion_ruling_is_the_old_write_and_is_logged(monkeypatch, caplog):
    def broken(*a, **k):
        raise RuntimeError("ledger unreadable")
    monkeypatch.setattr(B, "promotion_ruling", broken)
    monkeypatch.setattr(B, "apply_promotion_ruling", broken)
    with caplog.at_level(logging.WARNING, logger=MB.logger.name):
        assert MB._promotion(12, {"status": "draft"}, {"status": "up_next"}, False) is None
        assert MB._apply_promotion(12, {"outcome": "owe"}, "why") is None
    assert "promotion ruling failed and was skipped (RuntimeError" in caplog.text
    assert "apply promotion ruling failed and was skipped (RuntimeError" in caplog.text


def test_a_journal_that_cannot_be_written_costs_nothing(monkeypatch):
    def broken(*a, **k):
        raise OSError("read-only file system")
    monkeypatch.setattr(MB.SIM, "log_decision", broken)
    assert MB._swallowed("dedupe", ValueError("x"), "kept", journal=True) == "kept"
